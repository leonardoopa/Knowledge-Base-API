import os 
import dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class LLMBrain:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key
        )
        
        self.db_dir = "./chroma_db"
        self.vector_db = Chroma(
            persist_directory=self.db_dir, 
            embedding_function=self.embeddings
        )

    async def process_text(self, text: str):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        batch_size = 5
        total_chunks = len(chunks)
        print(f"--- Iniciando processamento de {total_chunks} pedaços ---")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            print(f"Processando lote {i} a {i + len(batch)}...")
            
            self.vector_db.add_texts(batch)
            
            time.sleep(1)
            
        print("--- Processamento concluído! ---")

    async def answer_question(self, question: str):
        docs = self.vector_db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        template = f"""Você é um assistente útil.
        Use o seguinte contexto para responder à pergunta.
        
        Contexto:
        {context}
        
        Pergunta: {question}
        """
        
        response = await self.llm.ainvoke(template)
        return response.content