from utils import WhisperAPI, OllamaAPI, uploadToServer, recordDetectSilenci, record, TTSKokoroESP, TTSGoogle
import json
import time
import os
import platform


idioma = "es"

if idioma == "es":
    nomAsistent = "Asistente: "
elif idioma == "en":
    nomAsistent = "Assistant: "

# Inicialitzar classe amb els següents paràmetres
ollama = OllamaAPI(temp=0.5, top_p=0.9, max_tokens=350, model='llama3.2', api_url='http://100.73.165.91:11434')

# Inicialitzar whisper amb els següents paràmetres
whisper = WhisperAPI(tasca='transcribe', idioma=idioma, batch_size=64, urlarxiu="http://100.73.165.91:4555/so/tmp.wav")


history = []

def historialText(history, promptText, language):
    # Funcio per afegir prompt al historial i per tenir memòria de la sessió
    history_str = " ".join(history)
    
    if language == "en":
        system_prompt = "System: You are a helpful assistant, you will assist User in their tasks."
        prompt = f"{system_prompt} User: {history_str} User: {promptText} "
    elif language == "es":
        system_prompt = "Sistema: Eres un asistente de IA, ayudarás a Usuario en sus tareas."
        prompt = f"{system_prompt} Usuario: {history_str} Usuario: {promptText} "
        
    return prompt


while True:
    # Gravar amb detecció de silenci - Procés
    gravacio = recordDetectSilenci()

    # Funció per pujar arxiu al servidor HTTP
    uploadToServer("tmp/tmp.wav")

    # Transcriure amb Whisper - Procés, extreure JSON i mostrar text
    outputSTT = whisper.transcribe_whisper()['output']['text']
    print(outputSTT)

    # Generar amb Ollama - Procés
    respostaText = ollama.generate(historialText(history=history, promptText=outputSTT, language=idioma))['message']['content']
    print(respostaText)
    
    print("Generant audio...")
    
    if(idioma == "es"):
        TTSKokoroESP(text=respostaText)
    else:
        TTSGoogle(text=respostaText, lang=idioma)
    
    #Afegir text previ al historial (per tenir memòria de la sessió)
    history.append(outputSTT + nomAsistent + respostaText)
