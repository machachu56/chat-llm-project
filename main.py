from utils import OllamaAPI


# Inicialitzar classe amb els següents paràmetres

ollama = OllamaAPI(temp=0.5, top_p=0.9, max_tokens=150, model='deepseek-r1:14b', api_url='http://100.73.165.91:11434')

# Generar amb Ollama - Procés
resposta = ollama.generate("What is 2 plus 2?")

print(resposta['message']['content'])