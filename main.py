from fastapi import FastAPI, UploadFile, File
import os 
import dotenv
from dotenv import load_dotenv

load_dotenv()

from brain import LLMBrain

app = FastAPI()
brain = LLMBrain()

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    await brain.process_text(text)

    return {"message": "Documento processado com sucesso."}


@app.post("/ask")
async def ask_question(question: str):
    answer = await brain.answer_question(question)
    return {"answer": answer}
