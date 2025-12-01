"""
🧠 NOVA RAG (Retrieval-Augmented Generation) Module

Advanced RAG system with:
- ChromaDB vector store
- Sentence transformers + custom NOVA embeddings
- Semantic search with re-ranking
- Document chunking and processing
- Conversation and knowledge memory
"""

from .embeddings import NovaEmbeddings, SentenceTransformerEmbeddings
from .vector_store import ChromaVectorStore
from .chunker import DocumentChunker
from .retriever import SemanticRetriever
from .memory import ConversationMemory, KnowledgeMemory
from .rag_pipeline import RAGPipeline

__all__ = [
    'NovaEmbeddings',
    'SentenceTransformerEmbeddings',
    'ChromaVectorStore',
    'DocumentChunker',
    'SemanticRetriever',
    'ConversationMemory',
    'KnowledgeMemory',
    'RAGPipeline',
]
