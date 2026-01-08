import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize vector store for document storage and retrieval
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="rahl_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """Setup embeddings model"""
        # Use sentence-transformers for local embeddings
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.embedding_model.encode(text).tolist()
    
    async def add_document(self, file_path: str, filename: str) -> str:
        """
        Process and add document to vector store
        
        Args:
            file_path: Path to document file
            filename: Original filename
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Load document based on file type
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path)
        elif file_ext == '.csv':
            loader = CSVLoader(file_path)
        elif file_ext in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            # Try text loader as fallback
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunks to vector store
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            embedding = self.get_embedding(chunk.page_content)
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[{
                    "source": filename,
                    "chunk_index": i,
                    "document_id": doc_id,
                    **chunk.metadata
                }]
            )
        
        return doc_id
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        documents = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                documents.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return documents
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection"""
        # Get all documents to extract unique source documents
        all_data = self.collection.get()
        if not all_data["metadatas"]:
            return []
        
        # Group by document_id
        docs_dict = {}
        for metadata in all_data["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id not in docs_dict:
                docs_dict[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata.get("source", "Unknown"),
                    "chunks_count": 0
                }
            docs_dict[doc_id]["chunks_count"] += 1
        
        return list(docs_dict.values())
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Success status
        """
        try:
            # Get all chunks for this document
            all_data = self.collection.get()
            chunks_to_delete = []
            
            for i, metadata in enumerate(all_data["metadatas"]):
                if metadata.get("document_id") == document_id:
                    chunks_to_delete.append(all_data["ids"][i])
            
            # Delete chunks
            if chunks_to_delete:
                self.collection.delete(ids=chunks_to_delete)
            
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_document_chunk_count(self, document_id: str) -> int:
        """Get number of chunks for a document"""
        all_data = self.collection.get()
        count = 0
        
        for metadata in all_data["metadatas"]:
            if metadata.get("document_id") == document_id:
                count += 1
        
        return count
