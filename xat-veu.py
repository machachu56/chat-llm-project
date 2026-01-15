from utils import STTAPI, OpenAILLMLocal, TTSChatterBox, uploadToServer, recordDetectSilenci, record, TTSKokoroESP, TTSGoogle
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
llm = OpenAILLMLocal(temp=0.5, top_p=0.9, max_tokens=350, model='qwen3:4b', api_url='http://192.168.99.2:8080/v1/')

# Inicialitzar whisper amb els següents paràmetres
stt = STTAPI(file_name="tmp/tmp.wav", api_url="http://192.168.99.2:5092/v1/")


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

    # Transcriure amb Whisper - Procés, extreure JSON i mostrar text
    outputSTT = stt.transcribe_stt()
    print(outputSTT)

    # Generar amb Ollama - Procés
    respostaText = llm.generate(historialText(history=history, promptText=outputSTT, language=idioma))
    print(respostaText)
    
    print("Generant audio...")
    
    if(idioma == "es"):
        #TTSKokoroESP(text=respostaText)
        TTSKokoroESP(api_url="http://192.168.99.2:4123/v1/", text=respostaText, voice="gladosesp")
    else:
        TTSGoogle(text=respostaText, lang=idioma)
    
    #Afegir text previ al historial (per tenir memòria de la sessió)
    history.append(outputSTT + nomAsistent + respostaText)
