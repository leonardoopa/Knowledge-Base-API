import requests

url = "http://127.0.0.1:8000/ask/stream"
payload = {
    "question": "Me resuma o documento que eu enviei.",
    "session_id": "leo_chat"
}

print("IA digitando: ", end="", flush=True)

# Faz a requisição avisando que é um Stream
with requests.post(url, json=payload, stream=True) as response:
    if response.status_code == 200:
        # Pega cada pedacinho (chunk) que o servidor manda e imprime na hora
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
    else:
        print(f"Erro: {response.status_code}")
        
print("\n\n[Fim da resposta]")