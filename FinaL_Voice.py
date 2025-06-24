# Enhanced 3D Dark Themed Voice Assistant with Fixed Message Display
import streamlit as st
import threading
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from pydub import AudioSegment
from pydub.playback import _play_with_pyaudio
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents, SpeakOptions
import sounddevice as sd
from queue import Queue

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

stt_active = threading.Event()
speaking = threading.Event()

# Deepgram client
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# LangChain setup
prompt_template = PromptTemplate.from_template(
    """You are a professional medical assistant. \
Your role is to assist users by answering medical questions, offering symptom-based advice, and suggesting general healthcare tips. \
You must respond in a **extremely concise**, accurate, and professional manner.

If the user asks about anything **non-medical** or outside your domain, politely respond:
\"I'm sorry, I am only trained to assist with medical-related topics.\"

Always stay within your medical assistant scope.

{history}
User: {input}
Assistant:"""
)
memory = ConversationBufferMemory(memory_key="history")
chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.6)
conversation = ConversationChain(llm=chat, memory=memory, prompt=prompt_template)

# Helpers
def dedupe_repeated_phrases(text):
    return " ".join(dict.fromkeys(text.split()))

def mic_stream(generator_queue, samplerate=16000, block_size=512):
    def callback(indata, frames, time, status):
        generator_queue.put(bytes(indata))
    with sd.RawInputStream(samplerate=samplerate, blocksize=block_size,
                           dtype='int16', channels=1, callback=callback):
        while stt_active.is_set():
            sd.sleep(50)

def deepgram_speak(text: str):
    speaking.set()
    st.session_state["ai_speaking"] = True
    filename = "tts_response.mp3"
    options = SpeakOptions(model="aura-2-thalia-en")
    deepgram_client.speak.rest.v("1").save(filename, {"text": text}, options)
    audio_segment = AudioSegment.from_file(filename, format="mp3")
    _play_with_pyaudio(audio_segment)
    st.session_state["ai_speaking"] = False
    speaking.clear()
    start_listening()

def speak_intro_and_start():
    speaking.set()
    st.session_state["ai_speaking"] = True
    intro_message = "Hello! I'm your medical chatbot. If you need any suggestions, let me know."
    filename = "tts_intro.mp3"
    options = SpeakOptions(model="aura-2-thalia-en")
    deepgram_client.speak.rest.v("1").save(filename, {"text": intro_message}, options)
    audio_segment = AudioSegment.from_file(filename, format="mp3")
    _play_with_pyaudio(audio_segment)
    st.session_state["ai_speaking"] = False
    speaking.clear()
    start_listening()

def start_listening():
    if speaking.is_set(): return
    stt_active.set()

    def deepgram_listener():
        generator_queue = Queue()
        threading.Thread(target=mic_stream, args=(generator_queue,), daemon=True).start()
        dg = deepgram_client.listen.live.v("1")

        def on_message(self, result, **kwargs):
            if not stt_active.is_set(): return
            transcript = result.channel.alternatives[0].transcript.strip()
            if not transcript: return
            stt_active.clear()
            cleaned = dedupe_repeated_phrases(transcript)
            st.session_state.user_said = cleaned
            st.session_state.chat_started = True
            response = conversation.predict(input=cleaned)
            st.session_state.ai_response = response
            threading.Thread(target=deepgram_speak, args=(response,), daemon=True).start()

        dg.on(LiveTranscriptionEvents.Transcript, on_message)
        options = LiveOptions(model="nova-3", sample_rate=16000, encoding="linear16")
        if not dg.start(options):
            print("Failed to start Deepgram STT")
            return

        while stt_active.is_set():
            data = generator_queue.get()
            dg.send(data)
        dg.finish()

    threading.Thread(target=deepgram_listener, daemon=True).start()

# UI config
st.set_page_config(page_title="3D Voice Medical Assistant", layout="centered")

# Init session state
for key in ["ai_speaking", "chat_started", "user_said", "ai_response"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "ai_speaking" else ""

# Custom styles
st.markdown("""
<style>
    body, html, .stApp {
        background: radial-gradient(circle at top, #0a0a0a, #001111);
        color: #e0f7fa !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        background-color: #000000 !important;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 30px #00ffe7;
    }
    .stButton button {
        background: linear-gradient(135deg, #00ffd5, #005050);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.75em 1.5em;
        box-shadow: 0 0 15px #00ffe7;
    }
    .avatar {
        width: 160px;
        height: 160px;
        margin: auto;
        background: radial-gradient(circle, #00f7ff 0%, #003f3f 100%);
        border-radius: 50%;
        box-shadow: 0 0 40px #00ffe7;
        animation: pulse 2s infinite;
    }
    .wave { display: flex; justify-content: center; gap: 6px; margin-top: 10px; }
    .wave div {
        width: 6px;
        height: 20px;
        background: #00ffff;
        animation: waveAnim 1s infinite ease-in-out;
    }
    .wave div:nth-child(2) { animation-delay: 0.1s; }
    .wave div:nth-child(3) { animation-delay: 0.2s; }
    .wave div:nth-child(4) { animation-delay: 0.3s; }
    .wave div:nth-child(5) { animation-delay: 0.4s; }

    @keyframes waveAnim { 0%, 100% { height: 20px; } 50% { height: 40px; } }
    @keyframes pulse { 0% { box-shadow: 0 0 20px #0ff; } 50% { box-shadow: 0 0 60px #0ff; } 100% { box-shadow: 0 0 20px #0ff; } }
    .chat-bubble {
        margin: 20px auto;
        padding: 15px 20px;
        border-radius: 15px;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 0 15px rgba(0,255,255,0.3);
        font-size: 18px;
    }
    .user-msg {
        background-color: rgba(0,255,255,0.1);
        color: #b2ebf2;
        border-left: 5px solid #00e5ff;
    }
    .ai-msg {
        background-color: rgba(0,255,255,0.2);
        color: #00fff0;
        border-right: 5px solid #00bcd4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="avatar"></div>', unsafe_allow_html=True)

if st.session_state.ai_speaking:
    st.markdown("<div class='wave'><div></div><div></div><div></div><div></div><div></div></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>🔊 Speaking...</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align:center;'>🟢 Listening...</p>", unsafe_allow_html=True)

# Auto-start chat
if not st.session_state.chat_started:
    if not st.session_state.get('_start_clicked'):
        st.session_state['_start_clicked'] = True
        threading.Thread(target=speak_intro_and_start).start()

# Chat bubbles
if st.session_state.user_said:
    st.markdown(f"<div class='chat-bubble user-msg'>🧑‍⚕️ <b>You said:</b><br>{st.session_state.user_said}</div>", unsafe_allow_html=True)

if st.session_state.ai_response:
    st.markdown(f"<div class='chat-bubble ai-msg'>🤖 <b>Assistant:</b><br>{st.session_state.ai_response}</div>", unsafe_allow_html=True)
