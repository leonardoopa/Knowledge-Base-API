import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega a chave do arquivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERRO: Chave GOOGLE_API_KEY não encontrada no .env")
else:
    print(f"--- Testando chave: {api_key[:5]}... ---")
    try:
        genai.configure(api_key=api_key)

        print("\nBuscando modelos de Embedding disponíveis...")
        found = False
        for m in genai.list_models():
            # Filtra apenas modelos que sabem fazer embedding
            if "embedContent" in m.supported_generation_methods:
                print(f"MODELO DISPONÍVEL: {m.name}")
                found = True

        if not found:
            print("Nenhum modelo de embedding encontrado para essa chave.")

    except Exception as e:
        print(f"Erro de conexão: {e}")
