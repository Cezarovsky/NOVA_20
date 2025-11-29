"""
Complete Transformer Architecture - Encoder and Decoder

This module implements the full Transformer architecture as described in
"Attention Is All You Need" (Vaswani et al., 2017), including:

1. Feed-Forward Networks (FFN)
2. Layer Normalization
3. Residual Connections
4. Transformer Encoder Layer
5. Transformer Decoder Layer
6. Complete Encoder Stack
7. Complete Decoder Stack
8. Full Transformer Model

Architecture Overview:

    Encoder:
        Input → Token + Position Embedding
          ↓
        N × [
          Multi-Head Self-Attention
          ↓
          Add & Norm (Residual + LayerNorm)
          ↓
          Feed-Forward Network
          ↓
          Add & Norm
        ]
          ↓
        Encoder Output

    Decoder:
        Target → Token + Position Embedding
          ↓
        N × [
          Masked Multi-Head Self-Attention
          ↓
          Add & Norm
          ↓
          Multi-Head Cross-Attention (with Encoder Output)
          ↓
          Add & Norm
          ↓
          Feed-Forward Network
          ↓
          Add & Norm
        ]
          ↓
        Linear → Softmax → Output Probabilities

Key Components:

Feed-Forward Network:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    or with GELU:
    FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    
    Typical dimensions:
    - d_model = 512
    - d_ff = 2048 (4× expansion)

Layer Normalization:
    LN(x) = γ * (x - μ) / (σ + ε) + β
    
    Where:
    - μ: mean over features
    - σ: std over features
    - γ, β: learnable scale and shift

Residual Connection:
    output = LayerNorm(x + Sublayer(x))
    
    Benefits:
    - Gradient flow (combat vanishing gradients)
    - Feature reuse (identity mapping)
    - Training stability

Usage:
    # Encoder-only (BERT-style)
    encoder = TransformerEncoder(
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000
    )
    output = encoder(input_ids, mask=mask)
    
    # Decoder-only (GPT-style)
    decoder = TransformerDecoder(
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_layers=12,
        d_ff=2048,
        max_len=5000,
        causal=True
    )
    output = decoder(input_ids)
    
    # Full Transformer (Translation)
    transformer = Transformer(
        src_vocab_size=30000,
        tgt_vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048
    )
    output = transformer(src_ids, tgt_ids, src_mask, tgt_mask)

Author: NOVA Development Team
Date: 28 November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

from src.ml.attention import MultiHeadAttention, CausalSelfAttention, create_padding_mask
from src.ml.embeddings import CombinedEmbedding


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Applies two linear transformations with activation in between,
    independently to each position (same across all positions).
    
    Mathematical Definition:
        FFN(x) = activation(xW₁ + b₁)W₂ + b₂
        
        Where:
        - x: input [batch, seq_len, d_model]
        - W₁: [d_model, d_ff]
        - W₂: [d_ff, d_model]
        - activation: ReLU or GELU
    
    Architecture:
        Input (d_model)
          ↓
        Linear (expand to d_ff)
          ↓
        Activation (ReLU/GELU)
          ↓
        Dropout
          ↓
        Linear (project back to d_model)
          ↓
        Dropout
          ↓
        Output (d_model)
    
    Why FFN?
        1. Non-linearity: Attention is linear, FFN adds non-linear capacity
        2. Position-wise: Each position processed independently (no interaction)
        3. Expansion: d_ff = 4×d_model provides more capacity
        4. Feature transformation: Learn complex feature combinations
    
    Activation Functions:
        - ReLU: max(0, x) - original Transformer, simple and fast
        - GELU: x * Φ(x) - modern choice, smoother, better gradients
          where Φ is Gaussian CDF
    
    Args:
        d_model: Model dimension (input/output)
        d_ff: Feed-forward dimension (hidden layer, typically 4×d_model)
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    
    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    
    Example:
        >>> ffn = FeedForwardNetwork(d_model=512, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize Feed-Forward Network
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (hidden size)
            dropout: Dropout probability
            activation: 'relu' or 'gelu'
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear layer: expand from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: project back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # First transformation with activation
        # [batch, seq_len, d_model] → [batch, seq_len, d_ff]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second transformation (project back)
        # [batch, seq_len, d_ff] → [batch, seq_len, d_model]
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Consists of:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual + layer norm)
    3. Feed-Forward Network
    4. Add & Norm
    
    Architecture:
        Input
          ↓
        Multi-Head Self-Attention
          ↓
        Add & Norm (x + attention(x))
          ↓
        Feed-Forward Network
          ↓
        Add & Norm (x + ffn(x))
          ↓
        Output
    
    Mathematical Flow:
        # Self-attention block
        attn_output = MultiHeadAttention(x, x, x, mask)
        x = LayerNorm(x + attn_output)
        
        # Feed-forward block
        ffn_output = FFN(x)
        x = LayerNorm(x + ffn_output)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation ('relu' or 'gelu')
        norm_first: Pre-norm vs post-norm (True = pre-norm)
    
    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    
    Example:
        >>> layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> x = torch.randn(32, 10, 512)
        >>> output = layer(x, mask=None)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = False
    ):
        """
        Initialize Transformer Encoder Layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: FFN activation function
            norm_first: Whether to use pre-norm (True) or post-norm (False)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_first = norm_first
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len] or None
        """
        if self.norm_first:
            # Pre-norm: normalize before sublayer
            x_norm = self.norm1(x)
            attn_output, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask, return_attention)
            x = x + self.dropout1(attn_output)
            
            x_norm = self.norm2(x)
            ffn_output = self.ffn(x_norm)
            x = x + self.dropout2(ffn_output)
        else:
            # Post-norm: normalize after residual connection (original Transformer)
            attn_output, attn_weights = self.self_attn(x, x, x, mask, return_attention)
            x = self.norm1(x + self.dropout1(attn_output))
            
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
    Consists of:
    1. Masked Multi-Head Self-Attention (causal)
    2. Add & Norm
    3. Multi-Head Cross-Attention (with encoder output)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    
    Architecture:
        Input
          ↓
        Masked Self-Attention (causal)
          ↓
        Add & Norm
          ↓
        Cross-Attention (with encoder)
          ↓
        Add & Norm
          ↓
        Feed-Forward Network
          ↓
        Add & Norm
          ↓
        Output
    
    Mathematical Flow:
        # Masked self-attention (on target sequence)
        self_attn = MaskedMultiHeadAttention(x, x, x)
        x = LayerNorm(x + self_attn)
        
        # Cross-attention (attend to encoder output)
        cross_attn = MultiHeadAttention(x, encoder_out, encoder_out)
        x = LayerNorm(x + cross_attn)
        
        # Feed-forward
        ffn_out = FFN(x)
        x = LayerNorm(x + ffn_out)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation
        norm_first: Pre-norm vs post-norm
    
    Example:
        >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> tgt = torch.randn(32, 10, 512)  # Target sequence
        >>> memory = torch.randn(32, 20, 512)  # Encoder output
        >>> output = layer(tgt, memory)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = False
    ):
        """Initialize Transformer Decoder Layer"""
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_first = norm_first
        
        # Masked self-attention (causal)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention (attend to encoder output)
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization (3 for 3 sublayers)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass
        
        Args:
            tgt: Target sequence [batch, tgt_len, d_model]
            memory: Encoder output [batch, src_len, d_model]
            tgt_mask: Target mask (causal) [batch, tgt_len, tgt_len]
            memory_mask: Source mask [batch, tgt_len, src_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch, tgt_len, d_model]
            self_attn_weights: [batch, num_heads, tgt_len, tgt_len] or None
            cross_attn_weights: [batch, num_heads, tgt_len, src_len] or None
        """
        if self.norm_first:
            # Pre-norm
            # Self-attention
            tgt_norm = self.norm1(tgt)
            self_attn_output, self_attn_weights = self.self_attn(
                tgt_norm, tgt_norm, tgt_norm, tgt_mask, return_attention
            )
            tgt = tgt + self.dropout1(self_attn_output)
            
            # Cross-attention
            tgt_norm = self.norm2(tgt)
            cross_attn_output, cross_attn_weights = self.cross_attn(
                tgt_norm, memory, memory, memory_mask, return_attention
            )
            tgt = tgt + self.dropout2(cross_attn_output)
            
            # Feed-forward
            tgt_norm = self.norm3(tgt)
            ffn_output = self.ffn(tgt_norm)
            tgt = tgt + self.dropout3(ffn_output)
        else:
            # Post-norm (original)
            # Self-attention
            self_attn_output, self_attn_weights = self.self_attn(
                tgt, tgt, tgt, tgt_mask, return_attention
            )
            tgt = self.norm1(tgt + self.dropout1(self_attn_output))
            
            # Cross-attention
            cross_attn_output, cross_attn_weights = self.cross_attn(
                tgt, memory, memory, memory_mask, return_attention
            )
            tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
            
            # Feed-forward
            ffn_output = self.ffn(tgt)
            tgt = self.norm3(tgt + self.dropout3(ffn_output))
        
        return tgt, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder Stack
    
    Stacks N encoder layers with embedding at input.
    Used in BERT-style models (encoder-only architectures).
    
    Architecture:
        Token IDs
          ↓
        Token + Position Embeddings
          ↓
        Encoder Layer 1
          ↓
        Encoder Layer 2
          ↓
        ...
          ↓
        Encoder Layer N
          ↓
        Output
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        padding_idx: Padding token index
        activation: FFN activation
        norm_first: Pre-norm vs post-norm
        positional_encoding: 'sinusoidal' or 'learned'
    
    Example:
        >>> encoder = TransformerEncoder(
        ...     vocab_size=30000,
        ...     d_model=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     d_ff=2048
        ... )
        >>> input_ids = torch.randint(0, 30000, (32, 20))
        >>> output = encoder(input_ids)
        >>> print(output.shape)  # torch.Size([32, 20, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        activation: str = 'gelu',
        norm_first: bool = False,
        positional_encoding: str = 'sinusoidal'
    ):
        """Initialize Transformer Encoder"""
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer (token + positional)
        self.embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            padding_idx=padding_idx,
            positional_encoding=positional_encoding,
            dropout=dropout
        )
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (if using pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if norm_first else None
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Forward pass
        
        Args:
            src: Source token IDs [batch, src_len]
            mask: Source mask [batch, src_len, src_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch, src_len, d_model]
            attention_weights: List of attention weights per layer or None
        """
        # Embedding
        x = self.embedding(src)
        
        # Apply encoder layers
        attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask, return_attention)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Final norm (for pre-norm)
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        return x, attention_weights


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder Stack
    
    Can be used in two modes:
    1. With encoder output (encoder-decoder architecture like translation)
    2. Decoder-only (causal LM like GPT) - set use_encoder=False
    
    Architecture (with encoder):
        Token IDs
          ↓
        Token + Position Embeddings
          ↓
        Decoder Layer 1 (self-attn + cross-attn + ffn)
          ↓
        Decoder Layer 2
          ↓
        ...
          ↓
        Decoder Layer N
          ↓
        Linear (→ vocab_size)
          ↓
        Output Logits
    
    Architecture (decoder-only):
        Token IDs
          ↓
        Token + Position Embeddings
          ↓
        Decoder Layer 1 (causal self-attn + ffn, no cross-attn)
          ↓
        ...
          ↓
        Linear (→ vocab_size)
          ↓
        Output Logits
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        padding_idx: Padding token index
        activation: FFN activation
        norm_first: Pre-norm vs post-norm
        positional_encoding: 'sinusoidal' or 'learned'
        use_encoder: Whether to use encoder-decoder attention
        causal: Whether to use causal masking (for LM)
        tie_weights: Whether to tie embedding and output weights
    
    Example:
        >>> # GPT-style (decoder-only)
        >>> decoder = TransformerDecoder(
        ...     vocab_size=30000,
        ...     d_model=768,
        ...     num_heads=12,
        ...     num_layers=12,
        ...     use_encoder=False,
        ...     causal=True
        ... )
        >>> input_ids = torch.randint(0, 30000, (32, 20))
        >>> logits = decoder(input_ids)
        >>> print(logits.shape)  # torch.Size([32, 20, 30000])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        activation: str = 'gelu',
        norm_first: bool = False,
        positional_encoding: str = 'learned',
        use_encoder: bool = True,
        causal: bool = True,
        tie_weights: bool = True
    ):
        """Initialize Transformer Decoder"""
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_encoder = use_encoder
        self.causal = causal
        
        # Embedding layer
        self.embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            padding_idx=padding_idx,
            positional_encoding=positional_encoding,
            dropout=dropout
        )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (if pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if norm_first else None
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output)
        if tie_weights:
            self.output_projection.weight = self.embedding.get_token_embedding_weight()
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()  # True = masked
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[list], Optional[list]]:
        """
        Forward pass
        
        Args:
            tgt: Target token IDs [batch, tgt_len]
            memory: Encoder output [batch, src_len, d_model] (if use_encoder=True)
            tgt_mask: Target mask [batch, tgt_len, tgt_len]
            memory_mask: Source mask [batch, tgt_len, src_len]
            return_attention: Whether to return attention weights
        
        Returns:
            logits: [batch, tgt_len, vocab_size]
            self_attn_weights: List of self-attention weights per layer
            cross_attn_weights: List of cross-attention weights per layer
        """
        batch_size, seq_len = tgt.size()
        
        # Create causal mask if needed
        if self.causal and tgt_mask is None:
            tgt_mask = self._create_causal_mask(seq_len, tgt.device)
        
        # Embedding
        x = self.embedding(tgt)
        
        # Apply decoder layers
        self_attn_weights = [] if return_attention else None
        cross_attn_weights = [] if return_attention else None
        
        for layer in self.layers:
            if self.use_encoder:
                if memory is None:
                    raise ValueError("memory (encoder output) required when use_encoder=True")
                x, self_attn, cross_attn = layer(x, memory, tgt_mask, memory_mask, return_attention)
            else:
                # Decoder-only: skip cross-attention
                x, self_attn, _ = layer(x, x, tgt_mask, None, return_attention)
                cross_attn = None
            
            if return_attention:
                self_attn_weights.append(self_attn)
                if cross_attn is not None:
                    cross_attn_weights.append(cross_attn)
        
        # Final norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder)
    
    Full implementation of "Attention Is All You Need" architecture
    for sequence-to-sequence tasks like machine translation.
    
    Architecture:
        Source → Encoder → Memory
        Target → Decoder (with Memory) → Output
    
    Use Cases:
    - Machine Translation (EN → FR, etc.)
    - Text Summarization
    - Question Answering
    - Any seq2seq task
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        src_padding_idx: Source padding token index
        tgt_padding_idx: Target padding token index
        activation: FFN activation
        norm_first: Pre-norm vs post-norm
        tie_weights: Tie decoder embedding and output weights
    
    Example:
        >>> transformer = Transformer(
        ...     src_vocab_size=30000,
        ...     tgt_vocab_size=30000,
        ...     d_model=512,
        ...     num_heads=8,
        ...     num_encoder_layers=6,
        ...     num_decoder_layers=6
        ... )
        >>> src = torch.randint(0, 30000, (32, 20))
        >>> tgt = torch.randint(0, 30000, (32, 15))
        >>> output = transformer(src, tgt)
        >>> print(output.shape)  # torch.Size([32, 15, 30000])
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        src_padding_idx: Optional[int] = None,
        tgt_padding_idx: Optional[int] = None,
        activation: str = 'gelu',
        norm_first: bool = False,
        tie_weights: bool = True
    ):
        """Initialize Transformer"""
        super().__init__()
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=src_padding_idx,
            activation=activation,
            norm_first=norm_first,
            positional_encoding='sinusoidal'
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=tgt_padding_idx,
            activation=activation,
            norm_first=norm_first,
            positional_encoding='sinusoidal',
            use_encoder=True,
            causal=True,
            tie_weights=tie_weights
        )
        
        self.d_model = d_model
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tensor:
        """
        Forward pass
        
        Args:
            src: Source token IDs [batch, src_len]
            tgt: Target token IDs [batch, tgt_len]
            src_mask: Source mask [batch, src_len, src_len]
            tgt_mask: Target mask [batch, tgt_len, tgt_len]
            memory_mask: Memory mask [batch, tgt_len, src_len]
            return_attention: Whether to return attention weights
        
        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Encode source
        memory, encoder_attn = self.encoder(src, src_mask, return_attention)
        
        # Decode target
        logits, decoder_self_attn, decoder_cross_attn = self.decoder(
            tgt, memory, tgt_mask, memory_mask, return_attention
        )
        
        if return_attention:
            return logits, {
                'encoder': encoder_attn,
                'decoder_self': decoder_self_attn,
                'decoder_cross': decoder_cross_attn
            }
        
        return logits
    
    def forward_embeddings(
        self,
        embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tensor:
        """
        Forward pass with pre-computed embeddings (for AI2AI training).
        
        Skips embedding layer and directly processes embeddings through
        transformer blocks. Used for training with AI2AI protocol where
        Claude provides embeddings directly.
        
        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, d_model]
            attention_mask: Attention mask [batch, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len, emb_dim = embeddings.shape
        
        if emb_dim != self.d_model:
            raise ValueError(
                f"Embedding dimension {emb_dim} doesn't match model dimension {self.d_model}"
            )
        
        # Create causal mask if needed (for decoder-only mode)
        if attention_mask is None:
            # Full attention (no masking)
            mask = None
        else:
            # Convert mask from [batch, seq_len] to [batch, 1, seq_len, seq_len]
            # Create causal mask for autoregressive generation
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=embeddings.device),
                diagonal=1
            ).bool()
            
            # Combine with padding mask
            if attention_mask.dim() == 2:
                # [batch, seq_len] → [batch, 1, 1, seq_len]
                padding_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)
                # Broadcast and combine
                mask = padding_mask | causal_mask.unsqueeze(0)
            else:
                mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Pass through encoder/decoder blocks
        # For GPT-style (decoder-only): use decoder
        # For BERT-style (encoder-only): use encoder
        
        if hasattr(self, 'encoder') and hasattr(self, 'decoder'):
            # Full encoder-decoder
            # In this case, treat embeddings as encoder input
            # Pass through encoder blocks directly (skip embedding layer)
            output = embeddings
            
            # Apply encoder layers manually
            for layer in self.encoder.layers:
                output = layer(output, mask)
            
            return output
        
        elif hasattr(self, 'decoder'):
            # Decoder-only (GPT-style)
            output = embeddings
            
            # Apply decoder layers
            for layer in self.decoder.layers:
                output, _, _ = layer(output, memory=None, tgt_mask=mask, memory_mask=None)
            
            return output
        
        elif hasattr(self, 'encoder'):
            # Encoder-only (BERT-style)
            output = embeddings
            
            # Apply encoder layers
            for layer in self.encoder.layers:
                output = layer(output, mask)
            
            return output
        
        else:
            raise RuntimeError("Model must have encoder, decoder, or both")


if __name__ == "__main__":
    """Test Transformer components"""
    print("=" * 80)
    print("Testing Transformer Architecture")
    print("=" * 80)
    
    # Test 1: Feed-Forward Network
    print("\n" + "-" * 80)
    print("Test 1: Feed-Forward Network")
    print("-" * 80)
    
    ffn = FeedForwardNetwork(d_model=512, d_ff=2048, dropout=0.1)
    x = torch.randn(32, 10, 512)
    output = ffn(x)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in ffn.parameters()):,}")
    
    # Test 2: Encoder Layer
    print("\n" + "-" * 80)
    print("Test 2: Transformer Encoder Layer")
    print("-" * 80)
    
    enc_layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
    x = torch.randn(32, 10, 512)
    output, attn = enc_layer(x, return_attention=True)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention shape: {attn.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in enc_layer.parameters()):,}")
    
    # Test 3: Decoder Layer
    print("\n" + "-" * 80)
    print("Test 3: Transformer Decoder Layer")
    print("-" * 80)
    
    dec_layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)
    tgt = torch.randn(32, 10, 512)
    memory = torch.randn(32, 20, 512)
    output, self_attn, cross_attn = dec_layer(tgt, memory, return_attention=True)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Self-attention shape: {self_attn.shape}")
    print(f"✅ Cross-attention shape: {cross_attn.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in dec_layer.parameters()):,}")
    
    # Test 4: Complete Encoder
    print("\n" + "-" * 80)
    print("Test 4: Transformer Encoder (6 layers)")
    print("-" * 80)
    
    encoder = TransformerEncoder(
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    )
    src = torch.randint(0, 30000, (32, 20))
    output, attn_list = encoder(src, return_attention=True)
    print(f"✅ Input shape: {src.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Number of layers: {len(attn_list)}")
    print(f"✅ Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test 5: Complete Decoder (GPT-style)
    print("\n" + "-" * 80)
    print("Test 5: Transformer Decoder (GPT-style, 12 layers)")
    print("-" * 80)
    
    decoder = TransformerDecoder(
        vocab_size=30000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        use_encoder=False,
        causal=True
    )
    input_ids = torch.randint(0, 30000, (32, 20))
    logits, self_attn_list, _ = decoder(input_ids, return_attention=True)
    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Output logits shape: {logits.shape}")
    print(f"✅ Number of layers: {len(self_attn_list)}")
    print(f"✅ Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Test 6: Full Transformer (Encoder-Decoder)
    print("\n" + "-" * 80)
    print("Test 6: Complete Transformer (Encoder-Decoder)")
    print("-" * 80)
    
    transformer = Transformer(
        src_vocab_size=30000,
        tgt_vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048
    )
    src = torch.randint(0, 30000, (32, 20))
    tgt = torch.randint(0, 30000, (32, 15))
    output = transformer(src, tgt)
    print(f"✅ Source shape: {src.shape}")
    print(f"✅ Target shape: {tgt.shape}")
    print(f"✅ Output logits shape: {output.shape}")
    print(f"✅ Total parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Parameter breakdown
    print("\n" + "-" * 80)
    print("Parameter Breakdown")
    print("-" * 80)
    
    encoder_params = sum(p.numel() for p in transformer.encoder.parameters())
    decoder_params = sum(p.numel() for p in transformer.decoder.parameters())
    total_params = sum(p.numel() for p in transformer.parameters())
    
    print(f"  Encoder: {encoder_params:,} parameters")
    print(f"  Decoder: {decoder_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    print(f"  Size: ~{total_params * 4 / (1024**2):.1f} MB (fp32)")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
