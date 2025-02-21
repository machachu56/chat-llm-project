from ollama import chat, Client

client = Client(host='http://100.73.165.91:11434')


stream = client.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'What is 2 plus 2?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)