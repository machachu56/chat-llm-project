import pyaudio
import numpy as np
import wave
import os
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
from colorama import Fore, Style, init

def recordDetectSilenciSOTA(output_filename="tmp/tmp.wav", threshold=0.5, silence_limit=1.5, debug=False):
    """
    Graba audio usando la librería oficial silero-vad.
    """
    init()
    
    # Cargar el modelo (por defecto usa ONNX si está disponible y es muy ligero)
    model = load_silero_vad()

    # Configuración de Audio
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000 # Silero requiere 16000Hz o 8000Hz
    CHUNK = 512  # Tamaño de bloque recomendado

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    print(Fore.CYAN + "Model Silero carregat. Escoltant..." + Style.RESET_ALL)

    frames = []
    # Cálculo de cuántos chunks equivalen al tiempo de silencio límite
    num_silence_chunks = int(RATE / CHUNK * silence_limit)
    silent_chunks_count = 0
    has_spoken = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Convertir el chunk actual a tensor para el modelo
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # El modelo espera un tensor de torch
            chunk_tensor = torch.from_numpy(audio_float32)
            
            # Obtener la probabilidad de voz para este chunk específico
            # Usamos el método interno del modelo para tiempo real
            speech_prob = model(chunk_tensor, RATE).item()

            if debug:
                print(f"Prob: {speech_prob:.3f} | Silenci: {silent_chunks_count}/{num_silence_chunks}")

            if speech_prob > threshold:
                if not has_spoken:
                    print(Fore.GREEN + "Parlant..." + Style.RESET_ALL)
                has_spoken = True
                silent_chunks_count = 0
            else:
                if has_spoken:
                    silent_chunks_count += 1

            # Si se supera el límite de silencio después de haber hablado
            if has_spoken and silent_chunks_count > num_silence_chunks:
                print(Fore.YELLOW + "Silenci detectat." + Style.RESET_ALL)
                break

    except KeyboardInterrupt:
        print("\nGrabació interrompuda.")

    # Finalizar stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Guardar el archivo final
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Arxiu desat a: {output_filename}")

    # Opcional: Validar con la función de la documentación que mencionaste
    # Esto sirve para limpiar el audio si hubo ruido al principio/final
    # clean_audio_with_timestamps(output_filename, model)

def clean_audio_with_timestamps(filename, model):
    """
    Usa get_speech_timestamps para recortar silencios innecesarios del archivo guardado.
    """
    from silero_vad import read_audio, save_audio, collect_chunks
    wav = read_audio(filename)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    if speech_timestamps:
        save_audio(filename, collect_chunks(speech_timestamps, wav), sampling_rate=16000)
        print("Audio optimitzat (silencis eliminats).")