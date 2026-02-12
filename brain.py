import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

class LLMBrain:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash", 
            google_api_key=api_key
        )
        
        self.db_dir = "./chroma_db"
        self.vector_db = Chroma(
            persist_directory=self.db_dir, 
            embedding_function=self.embeddings
        )

        self.chat_history = {}

    async def process_text(self, text: str):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        batch_size = 5
        total_chunks = len(chunks)
        print(f"--- Iniciando ingestão de {total_chunks} pedaços ---")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            print(f"Indexando lote {i} a {i + len(batch)}...")
            
            self.vector_db.add_texts(batch)
            
            time.sleep(1)
            
        print("--- Ingestão concluída com sucesso! ---")

    async def answer_question(self, question: str, session_id: str = "default"):
        """Busca contexto, gerencia memória e gera resposta via LLM."""
        
        docs = self.vector_db.similarity_search(question, k=3)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        
        history_str = "\n".join(self.chat_history[session_id])

        prompt = f"""Você é um assistente técnico especializado.
        Responda à pergunta do usuário usando APENAS o contexto fornecido abaixo.
        Considere o histórico da conversa para entender pronomes (como "ele", "isso", "lá").

        HISTÓRICO DA CONVERSA:
        {history_str}

        CONTEXTO DOS DOCUMENTOS:
        {context}

        PERGUNTA ATUAL: {question}
        
        RESPOSTA:"""

        response = await self.llm.ainvoke(prompt)
        content = response.content

        self.chat_history[session_id].append(f"Usuário: {question}")
        self.chat_history[session_id].append(f"IA: {content}")

        if len(self.chat_history[session_id]) > 10:
            self.chat_history[session_id] = self.chat_history[session_id][-10:]

        return content