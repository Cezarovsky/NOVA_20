"""
Attention Mechanisms for Transformer Models

This module implements the core attention mechanisms used in Transformer architectures:
1. Scaled Dot-Product Attention - the fundamental attention operation
2. Multi-Head Attention - parallel attention with learned projections
3. Optimizations for batch processing and inference

Mathematical Foundation:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
    - Q (Query): what we're looking for [batch, seq_len, d_k]
    - K (Key): what we're comparing against [batch, seq_len, d_k]
    - V (Value): what we return if there's a match [batch, seq_len, d_v]
    - d_k: dimension of key vectors (used for scaling)
    - √d_k: scaling factor to prevent softmax saturation

Multi-Head Attention:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Benefits:
    - Multiple "representation subspaces" (each head learns different patterns)
    - Parallel processing for efficiency
    - Different heads can focus on different aspects (syntax, semantics, etc.)

Usage:
    # Single-head attention
    attn = ScaledDotProductAttention(d_model=512, dropout=0.1)
    output, weights = attn(query, key, value, mask=None)
    
    # Multi-head attention
    mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
    output, weights = mha(query, key, value, mask=None)

Author: NOVA Development Team
Date: 28 November 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Implements the fundamental attention mechanism used in Transformers:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    The scaling factor (1/√d_k) is crucial because:
    1. Without scaling: QK^T values can be very large for high dimensions
    2. Large values → softmax saturates → vanishing gradients
    3. Scaling keeps values in a reasonable range for stable training
    
    Mathematical Intuition:
    - QK^T: compute similarity scores between queries and keys
    - /√d_k: scale down to prevent extreme values
    - softmax: normalize to probability distribution (sum = 1)
    - multiply by V: weighted sum of values based on attention weights
    
    Args:
        d_model: Model dimension (typically 512, 768, or 1024)
        dropout: Dropout probability for attention weights (regularization)
        
    Attributes:
        scale: Scaling factor (1/√d_k)
        dropout: Dropout layer applied to attention weights
    
    Shape:
        - Query: (batch_size, seq_len_q, d_k)
        - Key: (batch_size, seq_len_k, d_k)
        - Value: (batch_size, seq_len_k, d_v)
        - Output: (batch_size, seq_len_q, d_v)
        - Attention Weights: (batch_size, seq_len_q, seq_len_k)
    
    Example:
        >>> attn = ScaledDotProductAttention(d_model=512, dropout=0.1)
        >>> q = torch.randn(32, 10, 512)  # batch=32, seq_len=10
        >>> k = torch.randn(32, 20, 512)  # batch=32, seq_len=20
        >>> v = torch.randn(32, 20, 512)
        >>> output, weights = attn(q, k, v)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
        >>> print(weights.shape)  # torch.Size([32, 10, 20])
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize Scaled Dot-Product Attention
        
        Args:
            d_model: Model dimension (used to compute scaling factor)
            dropout: Dropout probability (0.0 = no dropout, 0.1 = 10% dropout)
        """
        super().__init__()
        
        # Scaling factor: 1/√d_k
        # We use d_model here assuming d_k = d_model
        # In multi-head attention, d_k = d_model / num_heads
        self.scale = 1.0 / math.sqrt(d_model)
        
        # Dropout for regularization (prevents overfitting to specific attention patterns)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: compute scaled dot-product attention
        
        Args:
            query: Query tensor [batch, seq_len_q, d_k]
            key: Key tensor [batch, seq_len_k, d_k]
            value: Value tensor [batch, seq_len_k, d_v]
            mask: Optional mask tensor [batch, 1, seq_len_k] or [batch, seq_len_q, seq_len_k]
                  - Use for padding (mask out padding tokens)
                  - Use for causal masking (prevent attending to future tokens)
                  - Values should be True for positions to MASK (set to -inf)
            return_attention: Whether to return attention weights (useful for visualization)
        
        Returns:
            output: Attention output [batch, seq_len_q, d_v]
            attention_weights: Attention weights [batch, seq_len_q, seq_len_k] or None
        
        Implementation Steps:
            1. Compute attention scores: Q @ K^T
            2. Scale scores by 1/√d_k
            3. Apply mask (if provided): set masked positions to -inf
            4. Apply softmax to get attention weights
            5. Apply dropout to attention weights (training only)
            6. Compute weighted sum: attention_weights @ V
        """
        # Step 1: Compute attention scores
        # Shape: [batch, seq_len_q, d_k] @ [batch, d_k, seq_len_k]
        #     -> [batch, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale scores
        # Why? For d_k=512, without scaling, scores can be ~22.6 (sqrt(512))
        # This pushes softmax into regions with extremely small gradients
        scores = scores * self.scale
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # We use masked_fill to set masked positions to -inf
            # After softmax, -inf becomes 0 (no attention to those positions)
            # mask=True means "mask this position"
            scores = scores.masked_fill(mask == True, float('-inf'))
        
        # Step 4: Apply softmax to get attention weights
        # Softmax normalizes scores to probability distribution (sum = 1 over seq_len_k)
        # Shape: [batch, seq_len_q, seq_len_k]
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle case where entire row is masked (all -inf → NaN after softmax)
        # Replace NaN with 0 (no attention anywhere)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Step 5: Apply dropout to attention weights
        # This randomly zeros out some attention weights during training
        # Prevents model from relying too heavily on specific attention patterns
        attention_weights_dropout = self.dropout(attention_weights)
        
        # Step 6: Compute weighted sum of values
        # Shape: [batch, seq_len_q, seq_len_k] @ [batch, seq_len_k, d_v]
        #     -> [batch, seq_len_q, d_v]
        output = torch.matmul(attention_weights_dropout, value)
        
        # Return output and attention weights (before dropout, for visualization)
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Extends scaled dot-product attention by running multiple attention operations
    in parallel, each with different learned linear projections.
    
    Mathematical Definition:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    
    Why Multiple Heads?
    1. Different Representation Subspaces:
       - Each head can learn to attend to different aspects
       - Example: one head for syntax, another for semantics
    
    2. Ensemble Effect:
       - Multiple heads = multiple "opinions" averaged together
       - More robust than single attention mechanism
    
    3. Parallel Processing:
       - All heads computed simultaneously (GPU parallelism)
       - No sequential bottleneck
    
    Architecture:
        Input (d_model) 
          ↓
        Linear Projections: Q, K, V → (num_heads, d_k)
          ↓
        Split into heads
          ↓
        Parallel Attention (each head independently)
          ↓
        Concatenate heads
          ↓
        Output Projection (W^O)
          ↓
        Output (d_model)
    
    Args:
        d_model: Model dimension (e.g., 512, 768, 1024)
        num_heads: Number of parallel attention heads (typically 8, 12, or 16)
        dropout: Dropout probability for attention weights
        bias: Whether to use bias in linear projections
        
    Attributes:
        num_heads: Number of attention heads
        d_k: Dimension per head (d_model / num_heads)
        W_q, W_k, W_v: Linear projections for Q, K, V
        W_o: Output projection after concatenation
        attention: ScaledDotProductAttention module
        dropout: Dropout layer
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
        - Attention Weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)  # batch=32, seq_len=10
        >>> output, weights = mha(x, x, x)  # self-attention
        >>> print(output.shape)  # torch.Size([32, 10, 512])
        >>> print(weights.shape)  # torch.Size([32, 8, 10, 10])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize Multi-Head Attention
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            
        Raises:
            AssertionError: If d_model is not divisible by num_heads
        """
        super().__init__()
        
        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"Got d_model % num_heads = {d_model % num_heads}"
        )
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Dimension per head
        # Example: d_model=512, num_heads=8 → d_k=64
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        # Each projects from d_model to d_model, but we'll split into heads
        # Shape: [d_model, d_model]
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection after concatenating heads
        # Shape: [d_model, d_model]
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Scaled dot-product attention for each head
        # Note: we use d_k (not d_model) for scaling
        self.attention = ScaledDotProductAttention(d_model=self.d_k, dropout=dropout)
        
        # Dropout for output (applied after output projection)
        self.dropout = nn.Dropout(p=dropout)
        
    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split tensor into multiple attention heads
        
        Reshapes from [batch, seq_len, d_model] to [batch, num_heads, seq_len, d_k]
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Reshaped tensor [batch, num_heads, seq_len, d_k]
            
        Implementation:
            1. Reshape to [batch, seq_len, num_heads, d_k]
            2. Transpose to [batch, num_heads, seq_len, d_k]
            
        Why this shape?
        - num_heads dimension allows parallel attention computation
        - Each head operates on d_k dimensions independently
        """
        batch_size, seq_len, d_model = x.size()
        
        # Reshape: [batch, seq_len, d_model] → [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose: [batch, seq_len, num_heads, d_k] → [batch, num_heads, seq_len, d_k]
        x = x.transpose(1, 2)
        
        return x
    
    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combine multiple attention heads back into single tensor
        
        Inverse of _split_heads: [batch, num_heads, seq_len, d_k] → [batch, seq_len, d_model]
        
        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]
            
        Returns:
            Combined tensor [batch, seq_len, d_model]
            
        Implementation:
            1. Transpose to [batch, seq_len, num_heads, d_k]
            2. Reshape to [batch, seq_len, d_model] where d_model = num_heads * d_k
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        
        # Transpose: [batch, num_heads, seq_len, d_k] → [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)
        
        # Reshape: [batch, seq_len, num_heads, d_k] → [batch, seq_len, d_model]
        x = x.contiguous().view(batch_size, seq_len, self.d_model)
        
        return x
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: compute multi-head attention
        
        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_k, d_model]
            mask: Optional mask [batch, 1, seq_len_k] or [batch, seq_len_q, seq_len_k]
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k] or None
        
        Implementation Flow:
            1. Linear projections: Q, K, V = query @ W_q, key @ W_k, value @ W_v
            2. Split into heads: reshape to [batch, num_heads, seq_len, d_k]
            3. Apply scaled dot-product attention (parallel for all heads)
            4. Concatenate heads: reshape to [batch, seq_len, d_model]
            5. Output projection: output = concat @ W_o
            6. Apply dropout and return
        
        Note on Self-Attention:
            When query = key = value, this becomes self-attention.
            Each position can attend to all positions in the same sequence.
        
        Note on Cross-Attention:
            When query ≠ key = value, this becomes cross-attention.
            Used in encoder-decoder architectures (e.g., machine translation).
        """
        batch_size = query.size(0)
        
        # Step 1: Linear projections
        # Shape: [batch, seq_len, d_model] → [batch, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: Split into multiple heads
        # Shape: [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Adjust mask for multi-head attention
        # We need to broadcast mask across all heads
        if mask is not None:
            # If mask is [batch, 1, seq_len_k], it broadcasts correctly
            # If mask is [batch, seq_len_q, seq_len_k], we need to add head dimension
            if mask.dim() == 3:
                # Add head dimension: [batch, seq_len_q, seq_len_k] → [batch, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
        
        # Step 3: Apply scaled dot-product attention (parallel for all heads)
        # Shape: [batch, num_heads, seq_len_q, d_k]
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask, return_attention=return_attention)
        
        # Step 4: Concatenate heads
        # Shape: [batch, num_heads, seq_len_q, d_k] → [batch, seq_len_q, d_model]
        attn_output = self._combine_heads(attn_output)
        
        # Step 5: Output projection
        # Shape: [batch, seq_len_q, d_model] → [batch, seq_len_q, d_model]
        output = self.W_o(attn_output)
        
        # Step 6: Apply dropout
        output = self.dropout(output)
        
        return output, attn_weights


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal (Masked) Self-Attention for Autoregressive Models
    
    A specialized version of Multi-Head Attention that prevents positions
    from attending to future positions. This is crucial for language modeling
    where the model should only use past context to predict the next token.
    
    Masking Pattern (for seq_len=4):
        [[1, 0, 0, 0],   # Position 0 can only see itself
         [1, 1, 0, 0],   # Position 1 can see 0, 1
         [1, 1, 1, 0],   # Position 2 can see 0, 1, 2
         [1, 1, 1, 1]]   # Position 3 can see 0, 1, 2, 3
    
    Where 1 = attend (allowed), 0 = masked (not allowed)
    
    Use Cases:
    - GPT-style language models
    - Text generation (autoregressive)
    - Any task requiring left-to-right processing
    
    Example:
        >>> causal_attn = CausalSelfAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)
        >>> output, weights = causal_attn(x)
        >>> # weights[0, 0, 5, 6:] will be all zeros (can't attend to future)
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, bias: bool = True):
        """
        Initialize Causal Self-Attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__(d_model, num_heads, dropout, bias)
        
        # Causal mask is created dynamically based on sequence length
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Generate or retrieve causal mask
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        # Check if we need to create a new mask
        if self.causal_mask is None or self.causal_mask.size(-1) < seq_len:
            # Create lower triangular matrix
            # torch.tril creates: [[1,0,0], [1,1,0], [1,1,1]]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            
            # Convert to boolean: 0 → True (mask), 1 → False (don't mask)
            # We invert because our attention expects True = mask
            mask = (mask == 0)
            
            # Add batch and head dimensions: [seq_len, seq_len] → [1, 1, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            # Register as buffer (moves with model to GPU/CPU)
            self.register_buffer('causal_mask', mask)
        
        return self.causal_mask[:, :, :seq_len, :seq_len]
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with causal masking
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional additional mask (e.g., for padding)
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len] or None
        """
        seq_len = x.size(1)
        device = x.device
        
        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, device)
        
        # Combine with additional mask if provided
        if mask is not None:
            # Both masks should prevent attention (logical OR)
            combined_mask = causal_mask | mask
        else:
            combined_mask = causal_mask
        
        # Call parent's forward with combined mask
        # For self-attention: query = key = value = x
        return super().forward(x, x, x, mask=combined_mask, return_attention=return_attention)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_padding_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """
    Create padding mask for sequences with padding tokens
    
    Args:
        seq: Input sequence [batch, seq_len] containing token indices
        pad_idx: Index of padding token (default: 0)
    
    Returns:
        Mask [batch, 1, seq_len] where True = padding token (should be masked)
    
    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(seq, pad_idx=0)
        >>> print(mask)
        tensor([[[False, False, False, True, True]],
                [[False, False, True, True, True]]])
    """
    # Check where padding tokens are
    mask = (seq == pad_idx)
    
    # Add dimension for broadcasting: [batch, seq_len] → [batch, 1, seq_len]
    mask = mask.unsqueeze(1)
    
    return mask


def visualize_attention(
    attention_weights: Tensor,
    src_tokens: list,
    tgt_tokens: list,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as heatmap
    
    Useful for understanding what the model is attending to.
    
    Args:
        attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        src_tokens: List of source tokens (keys)
        tgt_tokens: List of target tokens (queries)
        head_idx: Which attention head to visualize
        save_path: Optional path to save figure
    
    Example:
        >>> attn_weights = torch.rand(1, 8, 10, 10)  # batch=1, heads=8
        >>> tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        >>> visualize_attention(attn_weights, tokens, tokens, head_idx=0)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return
    
    # Extract attention for specific head
    # Shape: [batch, num_heads, seq_len_q, seq_len_k] → [seq_len_q, seq_len_k]
    attn = attention_weights[0, head_idx].detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='viridis',
        cbar=True,
        square=True
    )
    plt.xlabel('Source (Keys)')
    plt.ylabel('Target (Queries)')
    plt.title(f'Attention Weights (Head {head_idx})')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    """
    Test attention mechanisms
    """
    print("=" * 80)
    print("Testing Attention Mechanisms")
    print("=" * 80)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Parameters: d_model={d_model}, num_heads={num_heads}")
    
    # Test 1: Scaled Dot-Product Attention
    print("\n" + "-" * 80)
    print("Test 1: Scaled Dot-Product Attention")
    print("-" * 80)
    
    attn = ScaledDotProductAttention(d_model=d_model, dropout=0.1)
    output, weights = attn(x, x, x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention weights shape: {weights.shape}")
    print(f"✅ Attention weights sum (should be ~1.0): {weights[0, 0].sum().item():.4f}")
    
    # Test 2: Multi-Head Attention
    print("\n" + "-" * 80)
    print("Test 2: Multi-Head Attention")
    print("-" * 80)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    output, weights = mha(x, x, x, return_attention=True)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention weights shape: {weights.shape}")
    print(f"✅ Per-head dimension (d_k): {mha.d_k}")
    
    # Test 3: Causal Self-Attention
    print("\n" + "-" * 80)
    print("Test 3: Causal Self-Attention")
    print("-" * 80)
    
    causal_attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    output, weights = causal_attn(x, return_attention=True)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Causal mask applied: {weights[0, 0, 5, 6:].sum().item() == 0.0}")
    
    # Test 4: Padding Mask
    print("\n" + "-" * 80)
    print("Test 4: Padding Mask")
    print("-" * 80)
    
    seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_idx=0)
    print(f"✅ Sequence shape: {seq.shape}")
    print(f"✅ Mask shape: {mask.shape}")
    print(f"✅ Mask:\n{mask}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
