import os
import time
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from base64 import b64encode
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

# Load environment variables
load_dotenv()

# Retrieve API keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ASSISTANT_ID_KEY = os.getenv('ASSISTANT_ID_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize API clients
deepgramClient = DeepgramClient(DEEPGRAM_API_KEY)
elevenLabsClient = ElevenLabs(api_key=ELEVENLABS_API_KEY)
openAiClient = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Predefined options for Deepgram
options = PrerecordedOptions(model="nova-2", smart_format=True)


def get_audio(text, voice="Charlie", model="eleven_turbo_v2"):
    try:
        audio_stream = elevenLabsClient.generate(text=text, voice=voice, model=model)
        audio_bytes = b"".join(list(audio_stream))
        return b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error generating audio: {e}", flush=True)
        return None


def get_transcript(audio):
    temp_filename = "temp_audio.ogg"
    try:
        if not hasattr(audio, 'save'):
            raise Exception("Incompatible audio object. Expected a FileStorage instance with a save method.")
        
        audio.save(temp_filename)

        if not os.path.exists(temp_filename):
            raise Exception(f"File {temp_filename} not saved.")

        with open(temp_filename, "rb") as file:
            buffer_data = file.read()

        payload = {"buffer": buffer_data}
        response = deepgramClient.listen.prerecorded.v("1").transcribe_file(payload, options)
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        print(f"Error generating transcript: {e}", flush=True)
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def wait_on_run(run, thread_id):
    while run.status in ["queued", "in_progress"]:
        run = openAiClient.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        time.sleep(0.5)
    return run


def submit_message(thread_id, user_message):
    openAiClient.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_message)
    run = openAiClient.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID_KEY)
    print(run)
    return wait_on_run(run, thread_id)


def get_response(thread_id):
    try:
        response = openAiClient.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
        return response.data[0].content[0].text.value
    except Exception as e:
        print(f"Error in get_response: {e}", flush=True)
        return None


def get_text_response(text, thread_id=None):
    try:
        if not thread_id:
            thread = openAiClient.beta.threads.create()
            thread_id = thread.id
        submit_message(thread_id, text)
        return [get_response(thread_id), thread_id]
    except Exception as e:
        print(f"Error in get_text_response: {e}", flush=True)
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    print(data, flush=True)
    output = get_text_response(data.get('data'), data.get('thread_id'))

    if not output:
        return jsonify({"response": False, "message": "Error processing data"})

    if not data.get('audio_response'):
        return jsonify({"response": True, "message": output[0], "thread_id": output[1]})

    audio_base64 = get_audio(output[0])
    if audio_base64:
        return jsonify({"response": True, "message": output[0], "thread_id": output[1], "audio": audio_base64})
    else:
        return jsonify({"response": False, "message": "Failed to generate audio"})


@app.route('/audio', methods=['POST'])
def get_data_audio():
    audio_file = request.files.get('audio')
    user_input = get_transcript(audio_file)

    if not user_input:
        return jsonify({"message": "Failed to generate transcript", "response": False})

    output = get_text_response(user_input, request.form.get('thread_id'))
    if not request.form.get('audio_response') == 'true':
        return jsonify({"response": True, "message": output[0], "thread_id": output[1]})

    audio_base64 = get_audio(output[0])
    if audio_base64:
        return jsonify({"response": True, "message": output[0], "thread_id": output[1], "audio": audio_base64})
    else:
        return jsonify({"response": False, "message": "Failed to generate audio"})


if __name__ == '__main__':
    app.run()
