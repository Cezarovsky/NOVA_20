"""
🎯 NOVA Embeddings Module

Provides multiple embedding strategies:
1. Sentence Transformers (multilingual, pre-trained)
2. Custom NOVA embeddings (using our transformer)
3. Hybrid approach (best of both)
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """
    Embeddings using pre-trained sentence transformers.
    Supports multilingual models for Romanian + English.
    """
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None
    ):
        """
        Initialize sentence transformer embeddings.
        
        Args:
            model_name: HuggingFace model name
                Options:
                - "paraphrase-multilingual-MiniLM-L12-v2" (best for multilingual)
                - "all-MiniLM-L6-v2" (fast, English only)
                - "all-mpnet-base-v2" (best quality, English only)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"✓ Sentence transformer loaded")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Device: {self.device}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings as numpy array (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding (1D array)
        """
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings (2D array)
        """
        return self.embed(documents)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )


class NovaEmbeddings:
    """
    Custom embeddings using NOVA's transformer model.
    Uses the encoder part for semantic representation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        pooling: str = 'mean'
    ):
        """
        Initialize NOVA embeddings.
        
        Args:
            model: NOVA Transformer model
            tokenizer: NOVA tokenizer
            device: Device to run on
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling = pooling
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension from model
        self.embedding_dim = model.d_model
        
        logger.info(f"✓ NOVA embeddings initialized")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Pooling: {pooling}")
        logger.info(f"  Device: {self.device}")
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool token embeddings into sentence embedding.
        
        Args:
            hidden_states: Token hidden states (batch, seq_len, hidden_dim)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Sentence embeddings (batch, hidden_dim)
        """
        if self.pooling == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling == 'max':
            # Max pooling
            return hidden_states.max(dim=1)[0]
        
        elif self.pooling == 'cls':
            # Use first token ([CLS] style)
            return hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        token_ids = [self.tokenizer.encode(text) for text in texts]
        
        # Pad sequences
        max_len = max(len(ids) for ids in token_ids)
        pad_id = self.tokenizer.token_to_id.get('<pad>', 0)
        
        padded_ids = []
        attention_masks = []
        for ids in token_ids:
            padding_len = max_len - len(ids)
            padded = ids + [pad_id] * padding_len
            mask = [1] * len(ids) + [0] * padding_len
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        
        # Generate embeddings
        with torch.no_grad():
            # Get encoder output (for encoder-decoder model)
            if hasattr(self.model, 'encoder'):
                hidden_states = self.model.encoder(input_ids)
            else:
                # For decoder-only models
                hidden_states = self.model(input_ids, input_ids)
            
            # Pool to sentence embeddings
            embeddings = self._pool_embeddings(hidden_states, attention_mask)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents)


class HybridEmbeddings:
    """
    Hybrid embeddings combining sentence transformers and NOVA embeddings.
    Uses weighted average or concatenation.
    """
    
    def __init__(
        self,
        sentence_embeddings: SentenceTransformerEmbeddings,
        nova_embeddings: NovaEmbeddings,
        weight_sentence: float = 0.7,
        weight_nova: float = 0.3,
        mode: str = 'weighted'  # 'weighted' or 'concat'
    ):
        """
        Initialize hybrid embeddings.
        
        Args:
            sentence_embeddings: Sentence transformer embeddings
            nova_embeddings: NOVA embeddings
            weight_sentence: Weight for sentence embeddings (if mode='weighted')
            weight_nova: Weight for NOVA embeddings (if mode='weighted')
            mode: Combination mode ('weighted' or 'concat')
        """
        self.sentence_embeddings = sentence_embeddings
        self.nova_embeddings = nova_embeddings
        self.weight_sentence = weight_sentence
        self.weight_nova = weight_nova
        self.mode = mode
        
        if mode == 'weighted':
            self.embedding_dim = sentence_embeddings.embedding_dim
        elif mode == 'concat':
            self.embedding_dim = (
                sentence_embeddings.embedding_dim + 
                nova_embeddings.embedding_dim
            )
        
        logger.info(f"✓ Hybrid embeddings initialized")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Sentence weight: {weight_sentence}")
        logger.info(f"  NOVA weight: {weight_nova}")
        logger.info(f"  Final dim: {self.embedding_dim}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate hybrid embeddings."""
        sentence_emb = self.sentence_embeddings.embed(texts)
        nova_emb = self.nova_embeddings.embed(texts)
        
        if self.mode == 'weighted':
            # Normalize both to same dimension
            if sentence_emb.shape[1] != nova_emb.shape[1]:
                # Simple dimensionality matching via truncation/padding
                target_dim = min(sentence_emb.shape[1], nova_emb.shape[1])
                sentence_emb = sentence_emb[:, :target_dim]
                nova_emb = nova_emb[:, :target_dim]
            
            # Weighted average
            hybrid = (
                self.weight_sentence * sentence_emb +
                self.weight_nova * nova_emb
            )
            
            # Normalize
            norms = np.linalg.norm(hybrid, axis=1, keepdims=True)
            hybrid = hybrid / np.clip(norms, 1e-9, None)
            
            return hybrid
        
        elif self.mode == 'concat':
            # Concatenate
            return np.concatenate([sentence_emb, nova_emb], axis=1)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents)


if __name__ == "__main__":
    # Demo
    print("🎯 NOVA Embeddings Demo\n")
    
    # Test sentence transformers
    print("Loading sentence transformer...")
    st_emb = SentenceTransformerEmbeddings()
    
    texts = [
        "NOVA is an intelligent AI assistant.",
        "NOVA poate vorbi în limba română.",
        "Machine learning is fascinating."
    ]
    
    embeddings = st_emb.embed(texts)
    print(f"\n✓ Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Embedding dim: {st_emb.embedding_dim}")
    
    # Test similarity
    sim = st_emb.similarity(embeddings[0], embeddings[1])
    print(f"\n  Similarity between text 1 and 2: {sim:.4f}")
    
    print("\n✓ Embeddings demo complete!")
