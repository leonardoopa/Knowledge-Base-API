from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    
class QueryResponse(BaseModel):
    answer: str
    session_id: str
