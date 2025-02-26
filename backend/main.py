import os
import shutil
import sqlite3
import configparser

from vector_store import vector_store
from graph import get_graph, ChatState
from langchain.schema import HumanMessage
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from logger import backend_logger as logger
from langchain_docling.loader import ExportType
from fastapi import FastAPI, File, Path, UploadFile, HTTPException
from models import UploadResponse, DeleteResponse, ChatRequest, ChatResponse

# Load configuration
config = configparser.ConfigParser()
config.read("../config/config.ini")

UPLOAD_DIR = config["directory"]["upload"]
SQLITE_URI = config["sqlite"]["uri"]

K = int(config["milvus"]["k"])
SEARCH_DEPTH = config["tavily"]["search_depth"]
INCLUDE_ANSWER = config.getboolean("tavily", "include_answer")
INCLUDE_RAW_CONTENT = config.getboolean("tavily", "include_raw_content")
INCLUDE_IMAGES = config.getboolean("tavily", "include_images")
MAX_RESULTS = config["tavily"]["max_results"]
MODEL_NAME = config["llm"]["name"]
MODEL_PROVIDER = config["llm"]["provider"]
TEMPERATURE = float(config["llm"]["temperature"])

EMBED_MODEL_NAME = config["embedding"]["name"]
MAX_TOKENS = config["docling"]["max_tokens"]

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI application
app = FastAPI()

# Initialize the graph with parameters
graph = get_graph(
    k=K, 
    search_depth=SEARCH_DEPTH, 
    include_answer=INCLUDE_ANSWER, 
    include_raw_content=INCLUDE_RAW_CONTENT, 
    include_images=INCLUDE_IMAGES, 
    max_result=MAX_RESULTS, 
    model_name=MODEL_NAME, 
    model_provider=MODEL_PROVIDER,
    temperature=TEMPERATURE, 
    sqlite_url=SQLITE_URI
)

@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file, processes it using DoclingLoader, and stores it in the vector database.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Initializing DoclingLoader for file: {file_path}")
        # Process the file using DoclingLoader
        loader = DoclingLoader(
            file_path=file_path, 
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=EMBED_MODEL_NAME, max_tokens=MAX_TOKENS)
        )
        logger.info("DoclingLoader initialized successfully.")

        # Logging before loading documents
        logger.info(f"Loading documents from: {file_path}")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")

        # Add documents to the vector store
        ids = await vector_store.aadd_documents(documents)
        logger.info(f"File {file.filename} uploaded successfully.")
        return UploadResponse(message="File uploaded successfully", file_name=file.filename, document_ids=ids)
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.delete("/api/v1/delete/{file_name}", response_model=DeleteResponse)
async def delete_file(file_name: str):
    """
    Deletes a file from the upload directory and removes its reference from the vector store.
    """
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    if not os.path.exists(file_path):
        logger.warning(f"File {file_name} not found for deletion.")
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        await vector_store.adelete(expr=f'source == "{file_path}"')
        logger.info(f"File {file_name} deleted successfully.")
        return DeleteResponse(message="File deleted successfully", file_name=file_name)
    
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/api/v1/files")
async def get_files():
    """
    Retrieves a list of uploaded files.
    """
    try:
        files = os.listdir(UPLOAD_DIR)
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching file list: {str(e)}")


@app.post("/api/v1/conversations/{thread_id}/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, thread_id: str = Path(..., description="Conversation Thread ID")):
    """
    Handles chat queries by interacting with the LangGraph model.
    """
    try:
        logger.info(request)
        query = request.query
        file_path = os.path.join(UPLOAD_DIR, request.file) if request.file != 'None' else ""

        state = ChatState(
            messages=[HumanMessage(query)], 
            file_path=file_path,  
            rag_context="", 
            web_context=""
        )
        config = {"configurable": {"thread_id": f"{thread_id}"}}
        response = graph.invoke(input=state, config=config)["messages"][-1].content

        logger.info(f"Chat response for thread {thread_id}: {response}")
        return ChatResponse(response=response)
    
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")


@app.get("/api/v1/conversations")
async def get_conversations():
    """
    Fetches all distinct conversation thread IDs stored in the SQLite database.
    """
    try:
        with sqlite3.connect(SQLITE_URI, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
            thread_ids = [tid[0] for tid in cursor.fetchall()]
        return {"threads": thread_ids} 
    except Exception as e:
        logger.error(f"Failed to fetch conversation threads: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@app.get("/api/v1/conversation/{thread_id}")
async def get_conversation_by_thread_id(thread_id: str):
    """
    Retrieves a conversation history by thread ID from LangGraph.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        messages = graph.get_state(config).values["messages"]
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Failed to fetch conversation {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")
