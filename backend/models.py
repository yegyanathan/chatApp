from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class UploadResponse(BaseModel):
    message: str
    file_name: str
    document_ids: list

class DeleteResponse(BaseModel):
    message: str
    file_name: str

class ChatRequest(BaseModel):   
    query: str
    file: str


class ChatResponse(BaseModel):
    response: str

class ChatState(MessagesState):
    file_path: str = ""
    rag_context: str = ""
    web_context: str = ""


class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")