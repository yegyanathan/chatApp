from pydantic import BaseModel
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
    enable_web_search: bool


class ChatResponse(BaseModel):
    response: str

class ChatState(MessagesState):
    file_path: str = ""
    enable_web_search: bool = False
    rag_context: str = ""
    web_context: str = ""