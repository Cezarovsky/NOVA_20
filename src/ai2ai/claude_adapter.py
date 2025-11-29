"""
Claude Adapter: Convert Claude's responses to AI2AI embeddings.

Since we can't access Claude's internal embeddings directly,
we use a hybrid approach:
1. Get text response from Claude
2. Use our own embedding model to convert to vectors
3. Package as AI2AI message for NOVA

This is a bridge until we have direct API access to Claude's embeddings.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from anthropic import Anthropic

from .protocol import AI2AIMessage, MessageType, TransferMode, KnowledgeTransfer
from ..ml.embeddings import TokenEmbedding


class ClaudeAdapter:
    """
    Adapter to convert Claude interactions to AI2AI protocol.
    
    Architecture:
    Claude (text) → Embedding model (vectors) → AI2AI protocol → NOVA
    
    Eventually we want: Claude (internal vectors) → AI2AI → NOVA
    But Anthropic API doesn't expose embeddings yet.
    """
    
    def __init__(
        self,
        api_key: str,
        embedding_model: Optional[TokenEmbedding] = None,
        embedding_dim: int = 768,
        model: str = "claude-3-haiku-20240307"
    ):
        """
        Initialize Claude adapter.
        
        Args:
            api_key: Anthropic API key
            embedding_model: Embedding model (or None to create default)
            embedding_dim: Embedding dimension
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.embedding_dim = embedding_dim
        
        # Embedding model for text→vector conversion
        if embedding_model is None:
            self.embedding_model = TokenEmbedding(
                vocab_size=50000,  # Will be updated from tokenizer
                embedding_dim=embedding_dim
            )
        else:
            self.embedding_model = embedding_model
    
    def query(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        as_ai2ai: bool = True
    ) -> AI2AIMessage | str:
        """
        Query Claude and optionally convert to AI2AI format.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature
            as_ai2ai: If True, return AI2AI message; if False, return text
            
        Returns:
            AI2AI message or text response
        """
        # Call Claude API
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system:
            kwargs["system"] = system
        
        response = self.client.messages.create(**kwargs)
        text_response = response.content[0].text
        
        if not as_ai2ai:
            return text_response
        
        # Convert to embeddings
        embeddings = self._text_to_embeddings(text_response)
        
        # Package as AI2AI message
        return AI2AIMessage(
            message_type=MessageType.RESPONSE,
            embeddings=embeddings,
            source_model=self.model,
            target_model="nova",
            transfer_mode=TransferMode.RAW,
            metadata={
                "original_text": text_response,
                "prompt": prompt,
                "system": system,
            }
        )
    
    def extract_knowledge(
        self,
        domain: str,
        concepts: List[str],
        examples: Optional[List[str]] = None
    ) -> KnowledgeTransfer:
        """
        Extract structured knowledge from Claude about specific domain.
        
        This is used during training to transfer knowledge to NOVA.
        
        Args:
            domain: Knowledge domain (e.g., "physics", "math")
            concepts: List of concepts to extract
            examples: Optional example contexts
            
        Returns:
            KnowledgeTransfer object ready for NOVA training
        """
        # Ask Claude to explain concepts
        concept_explanations = []
        
        system_prompt = f"""You are a knowledge extraction system.
Provide concise, technical explanations of concepts in {domain}.
Focus on precise definitions and relationships."""
        
        for concept in concepts:
            prompt = f"Explain the concept: {concept}"
            response = self.query(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,  # Low temp for consistency
                as_ai2ai=False
            )
            concept_explanations.append(response)
        
        # Convert explanations to embeddings
        concept_embeddings = self._batch_text_to_embeddings(concept_explanations)
        
        # Process examples if provided
        example_embeddings = None
        if examples:
            example_embeddings = self._batch_text_to_embeddings(examples)
        
        # Extract relationships (ask Claude about connections)
        relationships = self._extract_relationships(concepts, domain)
        
        return KnowledgeTransfer(
            concept_embeddings=concept_embeddings,
            concept_names=concepts,
            relationships=relationships,
            domain=domain,
            example_embeddings=example_embeddings,
            example_contexts=examples,
            confidence=0.9,  # High confidence from Claude
        )
    
    def process_corpus(
        self,
        corpus_texts: List[str],
        chunk_size: int = 512,
        batch_size: int = 16
    ) -> List[AI2AIMessage]:
        """
        Process a corpus of texts into AI2AI messages for training.
        
        Args:
            corpus_texts: List of text chunks
            chunk_size: Max tokens per chunk
            batch_size: Process in batches
            
        Returns:
            List of AI2AI messages for NOVA training
        """
        messages = []
        
        for i in range(0, len(corpus_texts), batch_size):
            batch = corpus_texts[i:i + batch_size]
            
            # Convert batch to embeddings
            batch_embeddings = self._batch_text_to_embeddings(batch)
            
            # Create AI2AI messages
            for j, embeddings in enumerate(batch_embeddings):
                message = AI2AIMessage(
                    message_type=MessageType.KNOWLEDGE,
                    embeddings=embeddings.unsqueeze(0),  # Add batch dim
                    source_model=self.model,
                    target_model="nova",
                    transfer_mode=TransferMode.COMPRESSED,
                    metadata={
                        "corpus_index": i + j,
                        "original_text": batch[j][:200],  # Preview only
                    }
                )
                messages.append(message)
        
        return messages
    
    def _text_to_embeddings(self, text: str) -> torch.Tensor:
        """Convert single text to embeddings."""
        # Tokenize (simplified - real tokenizer would be more sophisticated)
        # For now, use character-level or word-level
        tokens = self._simple_tokenize(text)
        
        # Convert to embeddings using our model
        token_ids = torch.tensor([tokens], dtype=torch.long)
        embeddings = self.embedding_model(token_ids)  # [1, seq_len, emb_dim]
        
        # Mean pool to get single embedding vector
        return embeddings.mean(dim=1).squeeze(0)  # [emb_dim]
    
    def _batch_text_to_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Convert batch of texts to embeddings."""
        embeddings_list = []
        
        for text in texts:
            emb = self._text_to_embeddings(text)
            embeddings_list.append(emb)
        
        return torch.stack(embeddings_list)  # [batch, emb_dim]
    
    def _simple_tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """
        Simple tokenization (placeholder).
        
        TODO: Replace with proper tokenizer (e.g., tiktoken, sentencepiece)
        """
        # Character-level for now
        tokens = [ord(c) % 50000 for c in text[:max_length]]
        return tokens
    
    def _extract_relationships(
        self, 
        concepts: List[str],
        domain: str
    ) -> torch.Tensor:
        """
        Extract relationship graph between concepts.
        
        Returns:
            Adjacency matrix [num_concepts, num_concepts]
        """
        n = len(concepts)
        relationships = torch.zeros((n, n))
        
        # Ask Claude about relationships
        system_prompt = f"""Analyze relationships between concepts in {domain}.
Rate relationship strength from 0 (unrelated) to 1 (strongly related)."""
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i == j:
                    relationships[i, j] = 1.0  # Self-relation
                elif i < j:  # Only compute upper triangle
                    prompt = f"""How related are these concepts?
Concept A: {concept1}
Concept B: {concept2}

Respond with only a number from 0.0 to 1.0"""
                    
                    try:
                        response = self.query(
                            prompt=prompt,
                            system=system_prompt,
                            temperature=0.1,
                            as_ai2ai=False,
                            max_tokens=10
                        )
                        
                        # Parse response
                        score = float(response.strip())
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                        
                        relationships[i, j] = score
                        relationships[j, i] = score  # Symmetric
                    except:
                        # Default to 0.5 if parsing fails
                        relationships[i, j] = 0.5
                        relationships[j, i] = 0.5
        
        return relationships
    
    def stream_knowledge(
        self,
        knowledge_stream: List[str],
        callback: callable
    ):
        """
        Stream knowledge processing with callback.
        
        Useful for real-time training where NOVA learns as Claude processes.
        
        Args:
            knowledge_stream: Stream of text chunks
            callback: Function called with each AI2AI message
        """
        for i, text in enumerate(knowledge_stream):
            # Convert to AI2AI message
            embeddings = self._text_to_embeddings(text)
            
            message = AI2AIMessage(
                message_type=MessageType.KNOWLEDGE,
                embeddings=embeddings.unsqueeze(0),
                source_model=self.model,
                target_model="nova",
                transfer_mode=TransferMode.COMPRESSED,
                metadata={
                    "stream_index": i,
                    "is_final": i == len(knowledge_stream) - 1,
                }
            )
            
            # Invoke callback with message
            callback(message)
