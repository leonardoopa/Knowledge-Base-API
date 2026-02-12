from fastapi import FastAPI, UploadFile, File, HTTPException
from brain import LLMBrain
from models.query_dtos import QueryRequest, QueryResponse

app = FastAPI(title="Knowledge Base RAG API", version="1.0.0")
brain = LLMBrain()

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Recebe um arquivo de texto, processa em chunks e salva no banco vetorial.
    """
    try:
        content = await file.read()
        text = content.decode("utf-8")
        await brain.process_text(text)
        
        return {
            "message": f"Arquivo '{file.filename}' processado com sucesso!",
            "size": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na ingestão: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Responde perguntas com base no contexto e no histórico da sessão.
    """
    try:
        answer = await brain.answer_question(
            question=request.question, 
            session_id=request.session_id
        )
        
        return {
            "answer": answer,
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração da resposta: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    DEBUG: Mostra o histórico de conversa que a IA tem na memória RAM.
    """
    history = brain.chat_history.get(session_id, [])
    return {"session_id": session_id, "history": history}
