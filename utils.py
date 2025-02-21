from ollama import chat, Client
import requests
import wave
from colorama import Fore, Back, Style, init
import sounddevice as sd
import numpy as np
import io
from gtts import gTTS
from pydub import AudioSegment
from openai import OpenAI
import pyaudio
import time

# Creació de classe per Ollama i parametres
class OllamaAPI:
    def __init__(self, model, context=4096, max_tokens = 1, temp=0.5, top_p=0.9, api_url='http://10.66.66.10:11434'):
        # URL amb port d'Ollama
        self.api_url = api_url
        # Temperatura: Marca la variabilitat del text generat
        self.temp = temp
        # Top_p: Marca la probabilitat de les paraules generades (més diversitat o menys)
        self.top_p = top_p
        # Model: Model LLM a utilitzar
        self.model = model
        # Màxim tokens a generar
        self.max_tokens = max_tokens
        # Context: Màxims tokens que el model pot recordar
        self.context = context
    
    # Mètode per generar text
    def generate(self, prompt):
        # Inicialització del client Ollama
        client = Client(host=self.api_url)
        # Input a enviar al model
        response = client.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
            },
        ],
        #Opcions/parametres a enviar al model
        options={
            "temperature": self.temp,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "num_ctx": self.context
        }
        )
        return response


# Funcio per pujar arxius al servidor
def uploadToServer(arxiu, host='100.73.165.91:4555'):
    with open(arxiu, 'rb') as f:
        arxiu = f.read()
        requests.put(headers={'Content-Type': 'application/json'}, url=f'http://{host}/so/tmp.wav', data=arxiu)
        f.close()

# Creació de classe per Whisper i parametres
class WhisperAPI:
    def __init__(self, urlarxiu, tasca, idioma, batch_size, api_url='http://100.73.165.91:9000'):
        # URL amb port del Whisper Docker que vaig trobar
        self.api_url = api_url
        # URL de l'arxiu a transcriure, no es pot pujar arxiu directament
        self.urlarxiu = urlarxiu
        # Tasca a realitzar (transcriure o traduir)
        self.tasca = tasca
        # Idioma de la gravació
        self.idioma = idioma
        # Batch size de la tasca (+ és més ràpid però més costós de VRAM)
        self.batch_size = batch_size
    
    # Transcriure amb Whisper
    def transcribe_whisper(self):
        response = requests.post(url=self.api_url, json={
            "url": self.urlarxiu,
            "task": self.tasca,
            "language": self.idioma,
            "batch_size": self.batch_size
        })
        return response.json()
    


def recordDetectSilenci(output_filename="tmp/tmp.wav", threshold=10, silence_limit=1.5, debug=False):
    """
    Records audio from the microphone until silence is detected and saves it to a WAV file.
    
    Parameters:
        output_filename (str): Name of the output WAV file.
        threshold (int): RMS threshold below which audio is considered silent.
        silence_limit (float): Duration in seconds of continuous silence to trigger stopping.
    """
    
    init()
    print(Fore.GREEN + "Gravant..." + Style.RESET_ALL)
    # Audio configuration
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1              # Mono recording
    RATE = 44100              # Sampling rate in Hz
    CHUNK = 1024              # Frames per buffer

    # Determine how many consecutive silent chunks are required to consider it silence
    silence_chunks = int(RATE / CHUNK * silence_limit)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    silent_chunk_count = 0
    # Use a list to create a sliding window for RMS smoothing
    window_size = 10
    rms_window = []

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Convert byte data to a NumPy array of type int16
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Calculate current RMS value using NumPy
            current_rms = np.sqrt(np.mean(audio_data**2))
            
            # Update the sliding window using a list
            if len(rms_window) < window_size:
                rms_window.append(current_rms)
            else:
                rms_window = rms_window[1:] + [current_rms]
            
            avg_rms = sum(rms_window) / len(rms_window)
            
            if debug == True:
                print(f"Current RMS: {current_rms:.2f}, Average RMS: {avg_rms:.2f}")
            
            # Increase silent counter if the average RMS is below the threshold
            if avg_rms < threshold:
                silent_chunk_count += 1
            else:
                silent_chunk_count = 0  # Reset counter if sound is detected
            
            # Stop recording if silence persists for the specified number of chunks
            if silent_chunk_count > silence_chunks:
                print("Silenci detectat, procedint.")
                break
    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    
    # Clean up: stop and close the stream, then terminate PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded frames as a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {output_filename}")
    


# Funció feta per ChatGPT
def record():
    print("Parla al micròfon")
    init(convert=True)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    OUTPUT_FILENAME = "tmp/tmp.wav"

    # Gravar amb PyAudio
    audio = pyaudio.PyAudio()

    # Començar a gravar
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(Fore.GREEN +  "Gravant...")
    frames = []

    # Llegir gravacio
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(Style.RESET_ALL + "S'ha acabat de gravar.")

    # Tancar canal de gravacio
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Guardar a arxiu
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
       
def readBytesWav(bytes):  
    # Gràcies ChatGPT
    if(type(bytes) == io.BytesIO):
        bytes = bytes.read()
        
    wav_file = wave.open(io.BytesIO(bytes), "rb")  # Load into memory
    data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    return data, wav_file

def readBytesMp3(mp3_bytes):
    # Gràcies ChatGPT
    audio = AudioSegment.from_file(mp3_bytes, format="mp3")
    wav_data = io.BytesIO()
    audio.export(wav_data, format="wav")
    wav_data.seek(0)
    return wav_data

def TTSEsp(text):
    json = {"text": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=f"http://100.73.165.91:5130/convert/tts", headers=headers, json=json)
    if response.status_code == 200:  
        # Llegir .wav 
        bytes, wav_file = readBytesWav(response.content)
        # Reproduir so
        sd.play(bytes, samplerate=wav_file.getframerate())
        # Congelar fins que acabi la reproducció
        sd.wait()
    else:
        print(response.status_code)

def TTSKokoroESP(text):
    client = OpenAI(
    base_url="http://100.73.165.91:7777/v1", api_key="x")
    player = pyaudio.PyAudio().open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=24000, 
    output=True
    )

    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="ef_dora",
        response_format="pcm",
        input=text
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            player.write(chunk)


def TTSGoogle(text, lang):
    fp = io.BytesIO()
    tts = gTTS(text=text, lang = lang, slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    bytes = readBytesMp3(fp)
    data, wav_file = readBytesWav(bytes)
    speed_factor = 1.2
    sd.play(data, samplerate=wav_file.getframerate() * speed_factor)
    sd.wait()
