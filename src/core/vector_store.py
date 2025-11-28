"""
Vector Store - ChromaDB Integration

This module provides a wrapper around ChromaDB for vector storage and retrieval.
Supports:
- Document embeddings storage
- Image embeddings storage
- Audio transcription embeddings
- Semantic similarity search
- Metadata filtering
- Collection management

Architecture:

    ┌─────────────────────┐
    │   VectorStore       │  ← High-level interface
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   ChromaDB Client   │  ← Vector database
    └─────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    ┌────▼───┐  ┌───▼────┐
    │Documents│  │ Images │  ← Collections
    └─────────┘  └────────┘

Usage:

    # Initialize
    vector_store = VectorStore()
    
    # Add documents
    vector_store.add_documents(
        texts=["Hello world", "AI is amazing"],
        metadatas=[{"source": "doc1"}, {"source": "doc2"}]
    )
    
    # Search
    results = vector_store.search(
        query="artificial intelligence",
        n_results=5,
        collection_name="documents"
    )
    
    # Search with filters
    results = vector_store.search(
        query="AI",
        where={"source": "doc1"},
        n_results=10
    )

Author: NOVA Development Team
Date: 28 November 2025
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector Store wrapper for ChromaDB
    
    Provides high-level interface for:
    - Storing document/image/audio embeddings
    - Semantic similarity search
    - Metadata filtering
    - Collection management
    
    Collections:
    - documents: Text document embeddings
    - images: Image embeddings (from CLIP/similar)
    - audio: Audio transcription embeddings
    
    Args:
        persist_directory: Directory to persist database
        embedding_model: Model for embeddings (default: from settings)
        collection_prefix: Prefix for collection names
    
    Example:
        >>> store = VectorStore()
        >>> store.add_documents(
        ...     texts=["AI is the future"],
        ...     metadatas=[{"source": "article1"}]
        ... )
        >>> results = store.search("artificial intelligence")
        >>> print(results[0]['text'])
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_prefix: str = ""
    ):
        """Initialize vector store"""
        self.settings = get_settings()
        
        # Persistence directory
        self.persist_directory = persist_directory or self.settings.CHROMA_PERSIST_DIRECTORY
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Embedding configuration
        self.embedding_model = embedding_model or self.settings.DEFAULT_EMBEDDING_MODEL
        self.embedding_dimension = self.settings.EMBEDDING_DIMENSION
        
        # Collection prefix (useful for multi-tenant or testing)
        self.collection_prefix = collection_prefix
        
        # Initialize ChromaDB client
        self._client = None
        self._embedding_function = None
        
        # Collection names
        self.documents_collection_name = self._get_collection_name(
            self.settings.CHROMA_COLLECTION_DOCUMENTS
        )
        self.images_collection_name = self._get_collection_name(
            self.settings.CHROMA_COLLECTION_IMAGES
        )
        self.audio_collection_name = self._get_collection_name(
            self.settings.CHROMA_COLLECTION_AUDIO
        )
        
        logger.info(
            f"Initialized VectorStore: persist_dir={self.persist_directory}, "
            f"model={self.embedding_model}"
        )
    
    def _get_collection_name(self, base_name: str) -> str:
        """Get prefixed collection name"""
        if self.collection_prefix:
            return f"{self.collection_prefix}_{base_name}"
        return base_name
    
    @property
    def client(self) -> chromadb.Client:
        """Lazy initialization of ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._client
    
    @property
    def embedding_function(self):
        """Lazy initialization of embedding function"""
        if self._embedding_function is None:
            # Use Mistral embeddings (configured in settings)
            # Note: For production, you'd use MistralAI API for embeddings
            # For now, we use default sentence-transformers as fallback
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        return self._embedding_function
    
    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Get existing collection or create new one
        
        Args:
            collection_name: Name of collection
            metadata: Collection metadata
        
        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata=metadata or {}
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            texts: List of text documents
            metadatas: List of metadata dicts (one per document)
            ids: List of document IDs (generated if None)
            collection_name: Collection to add to (default: documents)
        
        Returns:
            List of document IDs
        
        Example:
            >>> ids = store.add_documents(
            ...     texts=["AI is great", "ML is powerful"],
            ...     metadatas=[{"source": "a"}, {"source": "b"}]
            ... )
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadata exists
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Validate lengths
        if len(texts) != len(metadatas):
            raise ValueError(
                f"texts ({len(texts)}) and metadatas ({len(metadatas)}) must have same length"
            )
        if len(texts) != len(ids):
            raise ValueError(
                f"texts ({len(texts)}) and ids ({len(ids)}) must have same length"
            )
        
        # Add to collection
        try:
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to {collection_name}")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Add pre-computed embeddings to vector store
        
        Useful when you already have embeddings from external model.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of IDs
            documents: Optional list of original documents (for reference)
            collection_name: Collection name
        
        Returns:
            List of IDs
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
        
        # Ensure metadata exists
        if metadatas is None:
            metadatas = [{} for _ in embeddings]
        
        try:
            collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                documents=documents
            )
            logger.info(f"Added {len(embeddings)} embeddings to {collection_name}")
            return ids
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def search(
        self,
        query: Union[str, List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text or embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"source": "article1"})
            where_document: Document content filter (e.g., {"$contains": "AI"})
            collection_name: Collection to search
            include: What to include in results
        
        Returns:
            List of results with documents, metadatas, distances
        
        Example:
            >>> # Search by text
            >>> results = store.search("artificial intelligence", n_results=5)
            >>> 
            >>> # Search with metadata filter
            >>> results = store.search(
            ...     "AI",
            ...     where={"source": "paper"},
            ...     n_results=10
            ... )
            >>> 
            >>> # Search by embedding vector
            >>> embedding = [0.1, 0.2, ..., 0.5]  # 1024-dim vector
            >>> results = store.search(embedding, n_results=3)
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        try:
            # Search by text query
            if isinstance(query, str):
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include
                )
            # Search by embedding vector
            else:
                results = collection.query(
                    query_embeddings=[query],
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include
                )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                }
                
                if 'documents' in results and results['documents'][0]:
                    result['text'] = results['documents'][0][i]
                
                if 'metadatas' in results and results['metadatas'][0]:
                    result['metadata'] = results['metadatas'][0][i]
                
                if 'embeddings' in results and results['embeddings'][0]:
                    result['embedding'] = results['embeddings'][0][i]
                
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include: List[str] = ["documents", "metadatas", "embeddings"]
    ) -> List[Dict[str, Any]]:
        """
        Get documents by IDs
        
        Args:
            ids: List of document IDs
            collection_name: Collection name
            include: What to include
        
        Returns:
            List of documents
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        try:
            results = collection.get(ids=ids, include=include)
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'])):
                result = {'id': results['ids'][i]}
                
                if 'documents' in results and results['documents']:
                    result['text'] = results['documents'][i]
                
                if 'metadatas' in results and results['metadatas']:
                    result['metadata'] = results['metadatas'][i]
                
                if 'embeddings' in results and results['embeddings']:
                    result['embedding'] = results['embeddings'][i]
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Update existing documents
        
        Args:
            ids: Document IDs to update
            documents: New document texts
            metadatas: New metadata
            embeddings: New embeddings
            collection_name: Collection name
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        try:
            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"Updated {len(ids)} documents in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Delete documents by IDs or metadata filter
        
        Args:
            ids: Document IDs to delete
            where: Metadata filter for deletion
            collection_name: Collection name
        
        Example:
            >>> # Delete by IDs
            >>> store.delete_documents(ids=["id1", "id2"])
            >>> 
            >>> # Delete by metadata
            >>> store.delete_documents(where={"source": "old_data"})
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        
        try:
            collection.delete(ids=ids, where=where)
            logger.info(f"Deleted documents from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def count_documents(self, collection_name: Optional[str] = None) -> int:
        """
        Count documents in collection
        
        Args:
            collection_name: Collection name
        
        Returns:
            Number of documents
        """
        collection_name = collection_name or self.documents_collection_name
        collection = self.get_or_create_collection(collection_name)
        return collection.count()
    
    def list_collections(self) -> List[str]:
        """
        List all collections
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete entire collection
        
        Args:
            collection_name: Collection to delete
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    def reset(self) -> None:
        """
        Reset entire database (delete all collections)
        
        ⚠️  WARNING: This deletes ALL data!
        """
        try:
            self.client.reset()
            logger.warning("Database reset - all collections deleted")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary
        """
        collections = self.list_collections()
        
        stats = {
            'total_collections': len(collections),
            'collections': {},
            'persist_directory': self.persist_directory,
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension
        }
        
        for collection_name in collections:
            try:
                count = self.count_documents(collection_name)
                stats['collections'][collection_name] = {
                    'count': count
                }
            except Exception as e:
                logger.warning(f"Failed to get stats for {collection_name}: {e}")
        
        return stats
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"VectorStore(persist_dir={self.persist_directory}, "
            f"model={self.embedding_model}, "
            f"collections={len(self.list_collections())})"
        )


if __name__ == "__main__":
    """Test vector store"""
    print("=" * 80)
    print("Testing Vector Store (ChromaDB)")
    print("=" * 80)
    
    # Test 1: Initialize
    print("\n" + "-" * 80)
    print("Test 1: Initialize Vector Store")
    print("-" * 80)
    
    store = VectorStore(collection_prefix="test")
    print(f"✅ Initialized: {store}")
    print(f"✅ Persist directory: {store.persist_directory}")
    print(f"✅ Collections: {store.list_collections()}")
    
    # Test 2: Add documents
    print("\n" + "-" * 80)
    print("Test 2: Add Documents")
    print("-" * 80)
    
    texts = [
        "Artificial intelligence is transforming the world",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing enables AI to understand text",
        "Computer vision allows AI to see and interpret images"
    ]
    
    metadatas = [
        {"topic": "AI", "category": "general"},
        {"topic": "ML", "category": "technical"},
        {"topic": "DL", "category": "technical"},
        {"topic": "NLP", "category": "technical"},
        {"topic": "CV", "category": "technical"}
    ]
    
    ids = store.add_documents(texts=texts, metadatas=metadatas)
    print(f"✅ Added {len(ids)} documents")
    print(f"✅ Document count: {store.count_documents()}")
    
    # Test 3: Search
    print("\n" + "-" * 80)
    print("Test 3: Search Documents")
    print("-" * 80)
    
    results = store.search("neural networks and deep learning", n_results=3)
    print(f"✅ Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Text: {result['text'][:60]}...")
        print(f"    Distance: {result['distance']:.4f}")
        print(f"    Metadata: {result['metadata']}")
    
    # Test 4: Search with filter
    print("\n" + "-" * 80)
    print("Test 4: Search with Metadata Filter")
    print("-" * 80)
    
    results = store.search(
        "AI technology",
        where={"category": "technical"},
        n_results=5
    )
    print(f"✅ Found {len(results)} technical documents")
    
    # Test 5: Get by IDs
    print("\n" + "-" * 80)
    print("Test 5: Get Documents by IDs")
    print("-" * 80)
    
    docs = store.get_by_ids(ids[:2])
    print(f"✅ Retrieved {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc['text'][:50]}...")
    
    # Test 6: Update
    print("\n" + "-" * 80)
    print("Test 6: Update Documents")
    print("-" * 80)
    
    store.update_documents(
        ids=[ids[0]],
        metadatas=[{"topic": "AI", "category": "updated"}]
    )
    print("✅ Updated document metadata")
    
    # Test 7: Delete
    print("\n" + "-" * 80)
    print("Test 7: Delete Documents")
    print("-" * 80)
    
    store.delete_documents(ids=[ids[-1]])
    print(f"✅ Deleted 1 document")
    print(f"✅ Remaining documents: {store.count_documents()}")
    
    # Test 8: Stats
    print("\n" + "-" * 80)
    print("Test 8: Database Statistics")
    print("-" * 80)
    
    stats = store.get_stats()
    print(f"✅ Total collections: {stats['total_collections']}")
    for name, info in stats['collections'].items():
        print(f"  - {name}: {info['count']} documents")
    
    # Cleanup
    print("\n" + "-" * 80)
    print("Cleanup")
    print("-" * 80)
    
    store.delete_collection(store.documents_collection_name)
    print("✅ Deleted test collection")
    
    print("\n" + "=" * 80)
    print("Vector Store tests completed")
    print("=" * 80)
