import os
import time
from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from base64 import b64encode
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()  # This loads the environment variables from .env file

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ASSISTANT_ID_KEY = os.getenv('ASSISTANT_ID_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize APIs
deepgramClient = DeepgramClient(DEEPGRAM_API_KEY)
elevenLabsClient = ElevenLabs(api_key=ELEVENLABS_API_KEY)
openAiClient = OpenAI(api_key=OPENAI_API_KEY)
asAgentValue=True
options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
)

app = Flask(__name__)
CORS(app)


def get_audio(text, voice="Charlie", model="eleven_turbo_v2"):
    try:
        # Generate the audio using the client
        text = elevenLabsClient.generate(text=text, voice=voice, model=model)

        # Convert audio generatior to bytes
        audio_bytes = b"".join(list(text))

        # Encode the audio bytes to a base64 string
        audio_base64 = b64encode(audio_bytes).decode('utf-8')

        return audio_base64
    except Exception as e:
        print(f"Error generating audio get_audio: {e}")
        return None
    

def get_transcript(audio):
    temp_filename = "temp_audio.ogg"
    try:
        # Check if the audio object is a FileStorage instance and has a 'save' method
        if not hasattr(audio, 'save'):
            raise Exception("The audio object is not compatible. Expected a FileStorage instance with a save method.")

        # Save the audio file directly using the 'save' method of FileStorage
        audio.save(temp_filename)

        # Ensure the file was saved correctly
        if not os.path.exists(temp_filename):
            raise Exception(f"Failed to save the file {temp_filename}")

        with open(temp_filename, "rb") as file:
            buffer_data = file.read()

        # Proceed with transcription using Deepgram
        payload: FileSource = {"buffer": buffer_data}
        response = deepgramClient.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        return transcript
    except Exception as e:
        print(f"Error generating transcript get_transcript: {e}")
        return None
    finally:
        # Always attempt to remove the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def wait_on_run(run, thread_id):
    while run.status == "queued" or run.status == "in_progress":
        run = openAiClient.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

def submit_message(thread_id, user_message):
    openAiClient.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_message
    )
    run = openAiClient.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID_KEY,
    )
    wait_on_run(run, thread_id)
    return run

def get_response(thread_id):
    response = openAiClient.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
    return response.data[0].content[0].text.value

def get_text_response(text,thread_id=None):
    try:
        print(f"text: {text} thread_id: {thread_id}")
        if thread_id is None or not thread_id.strip():
            print(f"if thread_id == None or thread_id.isspace()")
            thread = openAiClient.beta.threads.create()
            thread_id = thread.id
        print(f"thread_id: {thread_id}")
        submit_message(thread_id, text)
        response = get_response(thread_id)
        output = [response,thread_id]
        return output
    except Exception as e:
        print(f"Error generating assistant reponse get_text_response: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text=data.get('data')
    thread_id=data.get('thread_id')
    audio_response=data.get('audio_response')
    user_input = text
    try:
        # Get response from openai assistant
        output = get_text_response(user_input,thread_id)
        
        if not audio_response:
            return jsonify({"response": True, "message": output[0], "thread_id": output[1]})
        
        # Get the audio content from elevenlabs
        audio_base64 = get_audio(output[0])

        if audio_base64:
            return jsonify({"response": True, "message": output[0], "thread_id": output[1], "audio": audio_base64})
        else:
            return jsonify({"response": False, "message": "Failed to generate audio"})

    except Exception as e:
        print(e)
        error_message = f'Error get_data: {str(e)}'
        return jsonify({"message":error_message,"response":False})


@app.route('/audio', methods=['POST'])
def get_data_audio():
    audio_file = request.files.get('audio')
    thread_id = request.form.get('thread_id')
    audio_response = request.form.get('audio_response') == 'true'
    try:
        # get transcript from deepgram
        user_input = get_transcript(audio_file)
        
        if user_input == None or user_input.isspace():
            return jsonify({"message":"Failed to generate transcript","response":False})
        
        # get response from openai assistant
        output = get_text_response(user_input,thread_id)

        if not audio_response:
            return jsonify({"response": True, "message": output[0], "thread_id": output[1]})
        
        # Get the audio content from elevenlabs
        audio_base64 = get_audio(output[0])

        if audio_base64:
            return jsonify({"response": True, "message": output[0], "thread_id": output[1]})
        else:
            return jsonify({"response": False, "message": "Failed to generate audio"})

    except Exception as e:
        print(e)
        error_message = f'Error get_data_audio: {str(e)}'
        return jsonify({"message":error_message,"response":False})


if __name__ == '__main__':
    app.run()
