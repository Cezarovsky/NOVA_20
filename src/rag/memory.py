"""
🧠 NOVA Memory Management

Manages different types of memory for RAG:
- Conversation memory (short-term)
- Knowledge memory (long-term)
- Working memory (context window)
"""

from typing import List, Dict, Optional, Deque, Union
from collections import deque
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Short-term conversation memory.
    Stores recent chat history with sliding window.
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: Optional[int] = 2000
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to store
            max_tokens: Maximum total tokens (approximate)
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: Deque[Dict] = deque(maxlen=max_messages)
        
        logger.info(f"✓ Conversation memory initialized")
        logger.info(f"  Max messages: {max_messages}")
        logger.info(f"  Max tokens: {max_tokens}")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional metadata
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.messages.append(message)
        logger.debug(f"Added {role} message ({len(content)} chars)")
    
    def get_history(
        self,
        n_messages: Optional[int] = None,
        as_string: bool = False
    ) -> Union[List[Dict], str]:
        """
        Get conversation history.
        
        Args:
            n_messages: Number of recent messages (None = all)
            as_string: Return as formatted string
            
        Returns:
            List of messages or formatted string
        """
        messages = list(self.messages)
        
        if n_messages:
            messages = messages[-n_messages:]
        
        if as_string:
            lines = []
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                lines.append(f"{role}: {content}")
            return '\n'.join(lines)
        
        return messages
    
    def clear(self):
        """Clear all conversation history."""
        self.messages.clear()
        logger.info("Cleared conversation memory")
    
    def get_context_string(self, max_chars: int = 2000) -> str:
        """
        Get conversation context as string within character limit.
        
        Args:
            max_chars: Maximum characters
            
        Returns:
            Formatted context string
        """
        context = []
        total_chars = 0
        
        # Add messages from most recent backwards
        for msg in reversed(self.messages):
            msg_str = f"{msg['role']}: {msg['content']}\n"
            msg_len = len(msg_str)
            
            if total_chars + msg_len > max_chars:
                break
            
            context.insert(0, msg_str)
            total_chars += msg_len
        
        return ''.join(context)


class KnowledgeMemory:
    """
    Long-term knowledge memory using vector store.
    Stores facts, information, and learned content.
    """
    
    def __init__(
        self,
        vector_store,
        embeddings,
        chunker
    ):
        """
        Initialize knowledge memory.
        
        Args:
            vector_store: ChromaVectorStore instance
            embeddings: Embedding model
            chunker: DocumentChunker instance
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunker = chunker
        
        logger.info(f"✓ Knowledge memory initialized")
        logger.info(f"  Documents in store: {vector_store.count()}")
    
    def add_knowledge(
        self,
        content: str,
        source: str = "user_input",
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Add knowledge to long-term memory.
        
        Args:
            content: Knowledge content
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            List of document IDs
        """
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'source': source,
            'timestamp': datetime.now().isoformat()
        })
        
        # Chunk the content
        chunks = self.chunker.chunk_text(content, metadata)
        
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
        
        logger.info(f"✓ Added knowledge: {len(ids)} chunks")
        return ids
    
    def search_knowledge(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            n_results: Number of results
            filters: Metadata filters
            
        Returns:
            List of relevant knowledge chunks
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )
        
        # Format results
        knowledge = []
        for i in range(len(results['ids'])):
            knowledge.append({
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'score': 1.0 - results['distances'][i]
            })
        
        return knowledge
    
    def count(self) -> int:
        """Get total knowledge documents."""
        return self.vector_store.count()
    
    def clear(self):
        """Clear all knowledge."""
        self.vector_store.reset()
        logger.info("Cleared knowledge memory")


class WorkingMemory:
    """
    Working memory for current context.
    Combines conversation and retrieved knowledge into model context.
    """
    
    def __init__(
        self,
        max_context_length: int = 4000
    ):
        """
        Initialize working memory.
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
        self.current_context = ""
        
        logger.info(f"✓ Working memory initialized (max: {max_context_length} chars)")
    
    def build_context(
        self,
        query: str,
        conversation_history: str,
        retrieved_knowledge: List[Dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build complete context for model.
        
        Args:
            query: Current query
            conversation_history: Recent conversation
            retrieved_knowledge: Retrieved knowledge chunks
            system_prompt: Optional system prompt
            
        Returns:
            Complete context string
        """
        sections = []
        
        # System prompt
        if system_prompt:
            sections.append(f"System: {system_prompt}")
        
        # Retrieved knowledge
        if retrieved_knowledge:
            knowledge_texts = [k['text'] for k in retrieved_knowledge[:3]]
            sections.append("Knowledge Base:\n" + "\n".join(knowledge_texts))
        
        # Conversation history
        if conversation_history:
            sections.append(f"Conversation History:\n{conversation_history}")
        
        # Current query
        sections.append(f"Current Query: {query}")
        
        # Join and truncate if needed
        context = "\n\n".join(sections)
        
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
            logger.warning(f"Context truncated to {self.max_context_length} chars")
        
        self.current_context = context
        return context
    
    def get_context(self) -> str:
        """Get current working context."""
        return self.current_context
    
    def clear(self):
        """Clear working memory."""
        self.current_context = ""
        logger.info("Cleared working memory")


if __name__ == "__main__":
    # Demo
    print("🧠 Memory Management Demo\n")
    
    # Conversation memory
    conv_memory = ConversationMemory(max_messages=5)
    
    conv_memory.add_message("user", "Hello NOVA!")
    conv_memory.add_message("assistant", "Hello! How can I help you?")
    conv_memory.add_message("user", "Tell me about transformers.")
    
    print("Conversation History:")
    print(conv_memory.get_history(as_string=True))
    
    print("\n✓ Memory demo complete!")

