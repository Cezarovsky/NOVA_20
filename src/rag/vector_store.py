"""
🗄️ NOVA Vector Store using ChromaDB

Persistent vector database for semantic search with:
- ChromaDB for efficient similarity search
- Metadata filtering and management
- Collection persistence
- Batch operations
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Union, Any
import uuid
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store for NOVA's RAG system.
    
    Features:
    - Persistent storage on disk
    - Fast similarity search
    - Metadata filtering
    - Multiple collections support
    """
    
    def __init__(
        self,
        collection_name: str = "nova_knowledge",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (default: ./chroma_db)
            embedding_function: Custom embedding function (optional)
        """
        self.collection_name = collection_name
        
        # Setup persistence directory
        if persist_directory is None:
            persist_directory = str(Path.cwd() / "chroma_db")
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"✓ Loaded existing collection: {collection_name}")
            logger.info(f"  Documents: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"✓ Created new collection: {collection_name}")
    
    def add(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional)
            metadatas: List of metadata dicts (optional)
            ids: List of document IDs (optional, auto-generated if None)
            
        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Add to collection
        if embeddings is not None:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            # Let ChromaDB compute embeddings if function provided
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        logger.info(f"✓ Added {len(documents)} documents")
        return ids
    
    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "documentation"})
            where_document: Document content filter
            
        Returns:
            Dictionary with keys: 'ids', 'documents', 'metadatas', 'distances'
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Flatten results (ChromaDB returns nested lists)
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def search_by_text(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        Search using query text (requires embedding function).
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filter
            
        Returns:
            Dictionary with search results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Get documents by IDs or filter.
        
        Args:
            ids: List of document IDs
            where: Metadata filter
            limit: Maximum number of results
            
        Returns:
            Dictionary with documents and metadata
        """
        results = self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=['documents', 'metadatas']
        )
        
        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'metadatas': results['metadatas']
        }
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Update existing documents.
        
        Args:
            ids: List of document IDs to update
            documents: New document texts (optional)
            embeddings: New embeddings (optional)
            metadatas: New metadata (optional)
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"✓ Updated {len(ids)} documents")
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None
    ):
        """
        Delete documents by IDs or filter.
        
        Args:
            ids: List of document IDs to delete
            where: Metadata filter for deletion
        """
        self.collection.delete(
            ids=ids,
            where=where
        )
        logger.info(f"✓ Deleted documents")
    
    def count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()
    
    def reset(self):
        """Clear all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✓ Reset collection: {self.collection_name}")
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def peek(self, limit: int = 10) -> Dict[str, List]:
        """
        Peek at first few documents in collection.
        
        Args:
            limit: Number of documents to return
            
        Returns:
            Dictionary with documents and metadata
        """
        return self.collection.peek(limit=limit)


class MultiCollectionStore:
    """
    Manage multiple ChromaDB collections for different knowledge domains.
    
    Example use cases:
    - Separate collections for: documentation, conversations, code, etc.
    - Different collections per user or session
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize multi-collection store.
        
        Args:
            persist_directory: Directory to persist data
            embedding_function: Embedding function for all collections
        """
        self.persist_directory = persist_directory or str(Path.cwd() / "chroma_db")
        self.embedding_function = embedding_function
        self.collections: Dict[str, ChromaVectorStore] = {}
        
        logger.info(f"✓ Multi-collection store initialized")
    
    def get_collection(self, name: str) -> ChromaVectorStore:
        """
        Get or create a collection.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaVectorStore instance
        """
        if name not in self.collections:
            self.collections[name] = ChromaVectorStore(
                collection_name=name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        
        return self.collections[name]
    
    def search_all(
        self,
        query_embedding: Union[List[float], np.ndarray],
        n_results_per_collection: int = 3
    ) -> Dict[str, Dict[str, List]]:
        """
        Search across all collections.
        
        Args:
            query_embedding: Query embedding
            n_results_per_collection: Results per collection
            
        Returns:
            Dictionary mapping collection names to search results
        """
        results = {}
        for name, collection in self.collections.items():
            results[name] = collection.search(
                query_embedding=query_embedding,
                n_results=n_results_per_collection
            )
        return results


if __name__ == "__main__":
    # Demo
    print("🗄️ ChromaDB Vector Store Demo\n")
    
    # Create vector store
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    # Add some documents
    documents = [
        "NOVA is an intelligent AI assistant.",
        "NOVA uses transformer architecture for language understanding.",
        "The RAG system enhances NOVA with long-term memory.",
        "ChromaDB provides efficient vector storage and retrieval."
    ]
    
    print(f"Adding {len(documents)} documents...")
    # Note: For demo without embeddings, ChromaDB will use default
    ids = store.add(documents)
    print(f"✓ Added documents with IDs: {ids[:2]}...")
    
    # Count documents
    print(f"\nTotal documents: {store.count()}")
    
    # Peek at collection
    peek_results = store.peek(limit=2)
    print(f"\nFirst 2 documents:")
    for doc in peek_results['documents'][:2]:
        print(f"  - {doc[:50]}...")
    
    print("\n✓ Vector store demo complete!")
