import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print(f"{'ID':<5} {'Nom del Dispositiu'}")
    print("-" * 30)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Filtramos solo dispositivos que tengan canales de entrada
        if info['maxInputChannels'] > 0:
            print(f"{i:<5} {info['name']}")
    p.terminate()

# Puedes llamarla una vez para saber tu ID:
# list_audio_devices()

list_audio_devices()