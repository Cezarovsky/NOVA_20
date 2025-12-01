"""
🚀 NOVA RAG Pipeline

Complete RAG system integrating all components:
- Document processing and chunking
- Vector storage and retrieval  
- Memory management
- Context building for generation
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

from .embeddings import SentenceTransformerEmbeddings, NovaEmbeddings
from .vector_store import ChromaVectorStore
from .chunker import DocumentChunker
from .retriever import SemanticRetriever
from .memory import ConversationMemory, KnowledgeMemory, WorkingMemory

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for NOVA.
    
    Combines all RAG components into end-to-end system:
    1. Document ingestion and chunking
    2. Embedding generation
    3. Vector storage
    4. Semantic retrieval
    5. Context building
    6. Memory management
    """
    
    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "nova_knowledge",
        persist_directory: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_conversation_messages: int = 10
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Sentence transformer model name
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            max_conversation_messages: Max conversation history
        """
        logger.info("🚀 Initializing NOVA RAG Pipeline...")
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model
        )
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Initialize chunker
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy='smart'
        )
        
        # Initialize retriever
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            rerank=True,
            diversity_weight=0.3
        )
        
        # Initialize memories
        self.conversation_memory = ConversationMemory(
            max_messages=max_conversation_messages
        )
        
        self.knowledge_memory = KnowledgeMemory(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            chunker=self.chunker
        )
        
        self.working_memory = WorkingMemory(
            max_context_length=4000
        )
        
        logger.info("✓ RAG Pipeline initialized successfully!")
        logger.info(f"  Embedding model: {embedding_model}")
        logger.info(f"  Embedding dim: {self.embeddings.embedding_dim}")
        logger.info(f"  Documents in store: {self.vector_store.count()}")
    
    def add_document(
        self,
        text: str,
        source: str = "manual_input",
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Add a document to knowledge base.
        
        Args:
            text: Document text
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            List of chunk IDs
        """
        return self.knowledge_memory.add_knowledge(
            content=text,
            source=source,
            metadata=metadata
        )
    
    def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Add a file to knowledge base.
        
        Args:
            file_path: Path to file
            metadata: Additional metadata
            
        Returns:
            List of chunk IDs
        """
        # Chunk file
        chunks = self.chunker.chunk_file(file_path, metadata)
        
        # Extract texts and metadatas
        texts = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to vector store
        ids = self.vector_store.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        logger.info(f"✓ Added file: {file_path} ({len(ids)} chunks)")
        return ids
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        use_conversation_history: bool = True,
        return_sources: bool = True
    ) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            n_results: Number of knowledge chunks to retrieve
            use_conversation_history: Include conversation history in context
            return_sources: Include source documents in response
            
        Returns:
            Dictionary with 'context', 'sources', 'conversation_history'
        """
        # Retrieve relevant knowledge
        retrieved = self.retriever.retrieve(
            query=question,
            n_results=n_results
        )
        
        # Get conversation history
        conversation_history = ""
        if use_conversation_history:
            conversation_history = self.conversation_memory.get_context_string(
                max_chars=1000
            )
        
        # Build complete context
        context = self.working_memory.build_context(
            query=question,
            conversation_history=conversation_history,
            retrieved_knowledge=retrieved,
            system_prompt="You are NOVA, an intelligent AI assistant."
        )
        
        # Prepare response
        response = {
            'context': context,
            'query': question
        }
        
        if return_sources:
            response['sources'] = [
                {
                    'text': r['text'],
                    'source': r['metadata'].get('source', 'unknown'),
                    'score': r['score']
                }
                for r in retrieved
            ]
        
        if use_conversation_history:
            response['conversation_history'] = conversation_history
        
        logger.info(f"✓ Query processed: {len(retrieved)} sources retrieved")
        return response
    
    def chat(
        self,
        user_message: str,
        n_results: int = 3
    ) -> str:
        """
        Chat with NOVA using RAG.
        
        This method:
        1. Adds user message to conversation memory
        2. Queries knowledge base
        3. Builds context
        4. Returns context for generation
        
        Args:
            user_message: User's message
            n_results: Number of knowledge chunks
            
        Returns:
            Context string for model generation
        """
        # Add to conversation
        self.conversation_memory.add_message("user", user_message)
        
        # Query RAG
        rag_response = self.query(
            question=user_message,
            n_results=n_results,
            use_conversation_history=True,
            return_sources=False
        )
        
        return rag_response['context']
    
    def add_assistant_response(self, response: str):
        """
        Add assistant's response to conversation memory.
        
        Args:
            response: Assistant's response text
        """
        self.conversation_memory.add_message("assistant", response)
    
    def get_stats(self) -> Dict:
        """
        Get RAG system statistics.
        
        Returns:
            Dictionary with system stats
        """
        return {
            'total_documents': self.vector_store.count(),
            'conversation_messages': len(self.conversation_memory.messages),
            'embedding_dimension': self.embeddings.embedding_dim,
            'collections': self.vector_store.list_collections()
        }
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_memory.clear()
    
    def clear_knowledge(self):
        """Clear knowledge base (WARNING: destructive)."""
        self.knowledge_memory.clear()
    
    def reset(self):
        """Reset entire RAG system."""
        self.conversation_memory.clear()
        self.knowledge_memory.clear()
        self.working_memory.clear()
        logger.info("✓ RAG system reset")


if __name__ == "__main__":
    # Demo
    print("🚀 NOVA RAG Pipeline Demo\n")
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        collection_name="demo_collection",
        persist_directory="./demo_chroma_db"
    )
    
    # Add some knowledge
    print("\nAdding knowledge to RAG...")
    rag.add_document(
        "NOVA is an intelligent AI assistant built with transformer architecture.",
        source="about_nova"
    )
    rag.add_document(
        "The RAG system gives NOVA long-term memory and knowledge retrieval.",
        source="about_rag"
    )
    rag.add_document(
        "NOVA poate vorbi în limba română și engleză.",
        source="languages"
    )
    
    # Get stats
    stats = rag.get_stats()
    print(f"\nRAG Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    
    # Query
    print("\nQuerying RAG...")
    question = "What languages does NOVA support?"
    result = rag.query(question, n_results=2)
    
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. [{source['source']}] {source['text'][:100]}... (score: {source['score']:.3f})")
    
    print("\n✓ RAG Pipeline demo complete!")
