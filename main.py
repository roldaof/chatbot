import os
from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
from langchain_openai import OpenAI
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
#from langchain.chains import ConversationChain
#from langchain.memory import ConversationSummaryBufferMemory
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
asAgentValue=True
assistant = OpenAIAssistantRunnable(assistant_id=ASSISTANT_ID_KEY,asAgent=asAgentValue)
llm = OpenAI()
#memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
)

app = Flask(__name__)
CORS(app)


def get_audio(text, voice="Charlie", model="eleven_turbo_v2"):
    try:
        # Generate the audio using the client
        audio = ELEVENLABS_API_KEY.generate(text=text, voice=voice, model=model)

        # Convert audio generatior to bytes
        audio_bytes = b"".join(list(audio))

        # Encode the audio bytes to a base64 string
        audio_base64 = b64encode(audio_bytes).decode('utf-8')

        return audio_base64
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None
    

def get_transcript(audio):
    try:
        # Save the file temporarily
        temp_filename = "temp_audio.ogg"
        audio.save(temp_filename)

        with open(temp_filename, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgramClient.listen.prerecorded.v("1").transcribe_file(payload, options)

        # STEP 4: Print the response
        print(response.to_json(indent=4))

        # Now that the file is closed, it's safe to delete it
        os.remove(temp_filename)

        return response
    except Exception as e:
        os.remove(temp_filename)
        print(f"Error generating transcript: {e}")
        return None


def get_text_response(text):
    try:
        # conversation = ConversationChain(llm=llm,memory=memory)
        assistant_output = assistant.invoke({"content": text})
        # output = conversation.predict(input=user_input)
        # print(assistant_output[0])
        # print(assistant_output[0].content[0].text.value)

        # Extract the 'output' value
        output = assistant_output[0].content[0].text.value

        return output
    except Exception as e:
        print(f"Error generating assistant reponse: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text=data.get('data')
    user_input = text
    try:
        # Get response from openai assistant
        output = get_text_response(user_input)

        # Get the audio content from elevenlabs
        audio_base64 = get_audio(output)

        #memory.save_context({"input": user_input}, {"output": output})

        if audio_base64:
            return jsonify({"response": True, "message": output, "audio": audio_base64})
        else:
            return jsonify({"response": False, "message": "Failed to generate audio"})

    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})


@app.route('/audio', methods=['POST'])
def get_data_audio():
    audio_file = request.files['audio']
    try:
        print("get transcript")
        # get transcript from deepgram
        user_input = get_transcript(audio_file)
        
        if user_input == None:
            return jsonify({"message":"Failed to generate transcript","response":False})
        
        print("get response from openai assistant")
        # get response from openai assistant
        output = get_text_response(user_input)

        # Get the audio content from elevenlabs
        audio_base64 = get_audio(output)

        #memory.save_context({"input": user_input}, {"output": output})

        if audio_base64:
            return jsonify({"response": True, "message": output, "audio": audio_base64})
        else:
            return jsonify({"response": False, "message": "Failed to generate audio"})

    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})


if __name__ == '__main__':
    app.run()
