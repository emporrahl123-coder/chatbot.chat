import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

class RahlChatbot:
    def __init__(self, vector_store):
        """
        Initialize Rahl Chatbot
        
        Args:
            vector_store: Vector store instance for document retrieval
        """
        self.vector_store = vector_store
        self.setup_llm()
        
    def setup_llm(self):
        """Setup language model (OpenAI or local)"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=api_key
            )
        else:
            # Fallback to local model (for development)
            from langchain.llms import HuggingFacePipeline
            from transformers import pipeline
            
            # You can change this to any Hugging Face model
            generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                max_new_tokens=500
            )
            self.llm = HuggingFacePipeline(pipeline=generator)
    
    def chat(self, message: str, chat_history: List[dict] = None, use_rag: bool = True) -> Tuple[str, List[dict]]:
        """
        Process chat message with optional RAG
        
        Args:
            message: User message
            chat_history: Previous conversation history
            use_rag: Whether to use RAG or just LLM
            
        Returns:
            Tuple of (response, sources)
        """
        # Format chat history for context
        context = ""
        if chat_history:
            for msg in chat_history[-5:]:  # Last 5 messages for context
                role = "Human" if msg.get("role") == "user" else "Assistant"
                context += f"{role}: {msg.get('content', '')}\n"
        
        # Retrieve relevant documents if using RAG
        sources = []
        if use_rag:
            relevant_docs = self.vector_store.search(message, k=3)
            rag_context = "\n\n".join([doc["content"] for doc in relevant_docs])
            sources = [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                }
                for doc in relevant_docs
            ]
            
            # Create enhanced prompt
            prompt = f"""Context from documents:
{rag_context}

Previous conversation:
{context}

Human: {message}

Based on the provided context and conversation history, provide a helpful and accurate response. If the context doesn't contain relevant information, say so politely.

Assistant: """
        else:
            # Without RAG
            prompt = f"""Previous conversation:
{context}

Human: {message}

Assistant: """
        
        # Generate response
        try:
            response = self.llm.predict(prompt)
        except Exception as e:
            response = f"I apologize, but I encountered an error: {str(e)}"
        
        return response, sources
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.chat_history = []
