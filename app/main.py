import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import uvicorn

from app.chatbot import RahlChatbot
from app.vector_store import VectorStore

# Initialize FastAPI app
app = FastAPI(title="Rahl Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
chatbot = RahlChatbot(vector_store)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[dict]] = []
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []
    timestamp: str

class UploadResponse(BaseModel):
    filename: str
    document_id: str
    chunks_count: int

@app.get("/")
async def root():
    return {"message": "Rahl Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG"""
    try:
        response, sources = chatbot.chat(
            message=request.message,
            chat_history=request.chat_history,
            use_rag=request.use_rag
        )
        return ChatResponse(
            response=response,
            sources=sources,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process document
        document_id = await vector_store.add_document(tmp_path, file.filename)
        chunks_count = vector_store.get_document_chunk_count(document_id)
        
        # Clean up
        os.unlink(tmp_path)
        
        return UploadResponse(
            filename=file.filename,
            document_id=document_id,
            chunks_count=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = vector_store.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = vector_store.delete_document(document_id)
        return {"success": success, "document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
