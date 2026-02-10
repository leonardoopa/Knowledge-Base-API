import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class LLMBrain:
    def __init__(self):
        CANDIDATE_MODELS = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
        ]

        api_key = os.getenv("GOOGLE_API_KEY")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=api_key
        )

        self.llm = ChatGoogleGenerativeAI(
            model=next((model for model in CANDIDATE_MODELS if os.path.exists(model)), "models/gemini-2.5-flash"), 
            google_api_key=api_key
        )

        self.db_dir = "./chroma_db"
        self.vector_db = Chroma(
            persist_directory=self.db_dir, embedding_function=self.embeddings
        )

    async def process_text(self, text: str):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        batch_size = 5
        total_chunks = len(chunks)
        print(f"--- Iniciando processamento de {total_chunks} pedaços ---")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            print(f"Processando lote {i} a {i + len(batch)}...")

            self.vector_db.add_texts(batch)

            time.sleep(1)

        print("--- Processamento concluído! ---")

    async def answer_question(self, question: str):
        docs = self.vector_db.similarity_search(question, k=3)

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Você é um assistente técnico especializado que responde perguntas com base em documentos fornecidos.

        REGRAS IMPORTANTES:
        - Use APENAS o contexto abaixo para responder.
        - Se a informação não estiver no contexto, diga educadamente que não encontrou essa informação nos documentos.
        - Não tente inventar fatos ou usar conhecimentos externos.
        - Responda de forma direta e profissional.

        CONTEXTO RECUPERADO:
        {context}

        PERGUNTA DO USUÁRIO:
        {question}

        RESPOSTA:"""

        response = await self.llm.ainvoke(prompt)

        return response.content