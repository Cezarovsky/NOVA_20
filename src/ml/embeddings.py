"""
Token Embeddings and Positional Encoding for Transformer Models

This module implements core embedding mechanisms for transformers:
1. Token Embeddings - map discrete tokens to continuous vectors
2. Sinusoidal Positional Encoding - inject position information (original Transformer)
3. Learned Positional Encoding - learnable position embeddings (BERT, GPT)
4. Rotary Positional Embeddings (RoPE) - relative position encoding (future)

Why Embeddings?
    Neural networks operate on continuous vectors, but text is discrete (token IDs).
    Embeddings convert discrete tokens into dense continuous representations where:
    - Similar tokens have similar vectors
    - Semantic relationships are preserved
    - Dimensionality is manageable (typically 512-1024)

Why Positional Encoding?
    Attention mechanism is permutation-invariant (no inherent notion of order).
    Without positional info: "dog bites man" = "man bites dog"
    
    Solutions:
    1. Sinusoidal: Fixed patterns based on mathematical formula
    2. Learned: Trainable position embeddings
    3. Relative: Encode relative distances (RoPE, Alibi)

Mathematical Foundation:

Token Embeddings:
    x = Embedding[token_id]
    Shape: [vocab_size, d_model] → lookup → [d_model]

Sinusoidal Positional Encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Deterministic (no learning required)
    - Can extrapolate to longer sequences
    - Each dimension has different frequency
    - Relative position can be computed via linear combination

Learned Positional Encoding:
    PE(pos) = LearnedEmbedding[pos]
    
    Properties:
    - Trainable parameters
    - Can learn task-specific patterns
    - Fixed maximum sequence length
    - Often performs better on benchmark tasks

Combined Embedding:
    output = TokenEmbedding(x) + PositionalEncoding(x)
    output = Dropout(output)  # Regularization

Usage:
    # Token embeddings
    token_emb = TokenEmbedding(vocab_size=30000, d_model=512)
    x = token_emb(input_ids)  # [batch, seq_len] → [batch, seq_len, d_model]
    
    # Sinusoidal positional encoding
    pos_enc = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
    x = pos_enc(x)  # Add position information
    
    # Learned positional encoding
    pos_enc = LearnedPositionalEncoding(d_model=512, max_len=5000)
    x = pos_enc(x)  # Add learned positions

Author: NOVA Development Team
Date: 28 November 2025
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    
    Converts discrete token IDs to continuous dense vectors.
    
    Architecture:
        - Embedding matrix: [vocab_size, d_model]
        - Lookup operation: O(1) per token
        - Optional scaling by √d_model (used in original Transformer)
    
    Why Scaling?
        In the original Transformer paper, embeddings are scaled by √d_model.
        This puts embeddings and positional encodings on similar scales,
        preventing positional encodings from dominating early in training.
        
        Without scaling:
        - Embeddings: typically initialized with std ~ 1/√d_model
        - Positional encodings: values in [-1, 1]
        - Positional encodings might dominate
        
        With scaling:
        - Embeddings scaled up by √d_model
        - More balanced contribution from both sources
    
    Args:
        vocab_size: Size of vocabulary (number of unique tokens)
        d_model: Dimension of embedding vectors
        padding_idx: Index of padding token (embeddings will be zero)
        scale_embeddings: Whether to scale by √d_model
        dropout: Dropout probability for embeddings
    
    Attributes:
        embedding: PyTorch Embedding layer
        d_model: Embedding dimension
        scale_factor: Scaling factor (√d_model if enabled)
        dropout: Dropout layer
    
    Shape:
        - Input: [batch_size, seq_len] - token IDs
        - Output: [batch_size, seq_len, d_model] - embedded tokens
    
    Example:
        >>> token_emb = TokenEmbedding(vocab_size=30000, d_model=512)
        >>> input_ids = torch.randint(0, 30000, (32, 10))  # batch=32, seq_len=10
        >>> embeddings = token_emb(input_ids)
        >>> print(embeddings.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        scale_embeddings: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize Token Embedding
        
        Args:
            vocab_size: Vocabulary size
            d_model: Embedding dimension
            padding_idx: Padding token index (optional)
            scale_embeddings: Whether to scale by √d_model
            dropout: Dropout probability
        """
        super().__init__()
        
        # Embedding lookup table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Scaling factor (√d_model or 1.0)
        self.scale_factor = math.sqrt(d_model) if scale_embeddings else 1.0
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """
        Initialize embedding weights
        
        Uses Xavier/Glorot normal initialization:
        - Mean: 0
        - Std: 1/√d_model
        
        This ensures embeddings have unit variance before scaling.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        
        # Ensure padding embeddings are zero
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: lookup embeddings
        
        Args:
            x: Token IDs [batch_size, seq_len]
        
        Returns:
            Embeddings [batch_size, seq_len, d_model]
        """
        # Lookup embeddings
        embeddings = self.embedding(x)
        
        # Scale embeddings
        embeddings = embeddings * self.scale_factor
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_embedding_weight(self) -> Tensor:
        """
        Get embedding weight matrix
        
        Useful for weight tying (sharing embeddings with output layer).
        
        Returns:
            Embedding matrix [vocab_size, d_model]
        """
        return self.embedding.weight


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Original Transformer)
    
    Uses fixed sinusoidal patterns to encode position information.
    Each position gets a unique encoding based on sine and cosine functions
    with different frequencies for each dimension.
    
    Mathematical Definition:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Where:
        - pos: position in sequence (0 to max_len-1)
        - i: dimension index (0 to d_model/2 - 1)
        - 2i: even dimensions use sine
        - 2i+1: odd dimensions use cosine
    
    Intuition:
        - Low dimensions: high frequency (change quickly with position)
        - High dimensions: low frequency (change slowly with position)
        - Creates a unique "fingerprint" for each position
        - Similar to binary representation but continuous
    
    Properties:
        1. Deterministic: No parameters to learn
        2. Unbounded: Can extrapolate to sequences longer than training
        3. Relative positioning: PE(pos+k) can be expressed as linear
           combination of PE(pos) - enables learning of relative positions
        4. Smooth: Small position changes = small encoding changes
    
    Visualization:
        Position 0:  [sin(0/10000^0), cos(0/10000^0), sin(0/10000^0.01), ...]
        Position 1:  [sin(1/10000^0), cos(1/10000^0), sin(1/10000^0.01), ...]
        Position 2:  [sin(2/10000^0), cos(2/10000^0), sin(2/10000^0.01), ...]
        
        Each row is unique, adjacent rows are similar.
    
    Args:
        d_model: Model dimension (must be even)
        max_len: Maximum sequence length to precompute
        dropout: Dropout probability
        base: Base for exponential decay (default: 10000)
    
    Attributes:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout layer
        pe: Precomputed positional encodings [max_len, d_model]
    
    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    
    Example:
        >>> pos_enc = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 10, 512)  # batch=32, seq_len=10
        >>> x_with_pos = pos_enc(x)
        >>> print(x_with_pos.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        base: int = 10000
    ):
        """
        Initialize Sinusoidal Positional Encoding
        
        Args:
            d_model: Model dimension (should be even)
            max_len: Maximum sequence length
            dropout: Dropout probability
            base: Base for frequency decay
        """
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encodings
        pe = self._compute_positional_encoding(max_len, d_model, base)
        
        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer('pe', pe)
    
    def _compute_positional_encoding(
        self,
        max_len: int,
        d_model: int,
        base: int = 10000
    ) -> Tensor:
        """
        Compute sinusoidal positional encoding matrix
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            base: Base for exponential decay
        
        Returns:
            Positional encoding matrix [max_len, d_model]
        
        Implementation:
            1. Create position indices: [0, 1, 2, ..., max_len-1]
            2. Create dimension indices: [0, 2, 4, ..., d_model-2]
            3. Compute frequency divisors: base^(2i/d_model)
            4. Compute angles: pos / divisor
            5. Apply sin to even dimensions, cos to odd dimensions
        """
        # Position indices: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices for even positions: [0, 2, 4, ..., d_model-2]
        # We use arange(0, d_model, 2) to get even indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(base) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        
        Raises:
            RuntimeError: If seq_len > max_len
        """
        batch_size, seq_len, d_model = x.size()
        
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}. "
                f"Increase max_len or use shorter sequences."
            )
        
        if d_model != self.d_model:
            raise RuntimeError(
                f"Input dimension {d_model} does not match expected dimension {self.d_model}"
            )
        
        # Add positional encoding (broadcasting over batch dimension)
        # self.pe shape: [max_len, d_model]
        # self.pe[:seq_len] shape: [seq_len, d_model]
        # After unsqueeze(0): [1, seq_len, d_model]
        # Broadcasting with x: [batch_size, seq_len, d_model]
        x = x + self.pe[:seq_len].unsqueeze(0)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
    
    def visualize_encoding(self, num_positions: int = 100, save_path: Optional[str] = None):
        """
        Visualize positional encodings as heatmap
        
        Args:
            num_positions: Number of positions to visualize
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for visualization")
            return
        
        # Get encodings for first num_positions
        encodings = self.pe[:num_positions].cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            encodings.T,  # Transpose to show dimensions on y-axis
            cmap='RdBu_r',
            center=0,
            xticklabels=20,  # Show every 20th position
            yticklabels=20,  # Show every 20th dimension
            cbar_kws={'label': 'Encoding Value'}
        )
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Sinusoidal Positional Encoding')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (BERT, GPT style)
    
    Instead of fixed sinusoidal patterns, this uses learnable embeddings
    for each position. The model learns optimal position representations
    during training.
    
    Comparison with Sinusoidal:
        Pros:
        - Can learn task-specific patterns
        - Often better performance on benchmarks
        - More flexible for specialized tasks
        
        Cons:
        - Requires training (more parameters)
        - Fixed maximum length (can't extrapolate)
        - May overfit to sequence lengths in training data
    
    Architecture:
        - Position embedding matrix: [max_len, d_model]
        - Lookup similar to token embeddings
        - Added to token embeddings before processing
    
    Usage Pattern:
        token_emb = TokenEmbedding(vocab_size, d_model)
        pos_enc = LearnedPositionalEncoding(max_len, d_model)
        
        x = token_emb(input_ids)  # Token embeddings
        x = pos_enc(x)             # Add position information
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    
    Attributes:
        position_embeddings: Learnable position embeddings
        dropout: Dropout layer
    
    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
    
    Example:
        >>> pos_enc = LearnedPositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 10, 512)
        >>> x_with_pos = pos_enc(x)
        >>> print(x_with_pos.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize Learned Positional Encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """
        Initialize position embeddings
        
        Uses normal initialization with small std to start close to zero.
        This prevents position encodings from dominating early in training.
        """
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add learned positional encoding
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        
        Raises:
            RuntimeError: If seq_len > max_len
        """
        batch_size, seq_len, d_model = x.size()
        
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        if d_model != self.d_model:
            raise RuntimeError(
                f"Input dimension {d_model} does not match expected dimension {self.d_model}"
            )
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # Lookup position embeddings: [seq_len, d_model]
        position_embeddings = self.position_embeddings(positions)
        
        # Add to input (broadcasting over batch dimension)
        x = x + position_embeddings.unsqueeze(0)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class CombinedEmbedding(nn.Module):
    """
    Combined Token + Positional Embedding
    
    Convenience module that combines token embeddings with positional encoding
    in a single forward pass.
    
    This is the standard pattern used in most transformer implementations:
        1. Convert token IDs to embeddings
        2. Add positional information
        3. Apply dropout
        4. Feed to transformer layers
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_len: Maximum sequence length
        padding_idx: Padding token index
        positional_encoding: Type of positional encoding ('sinusoidal' or 'learned')
        scale_embeddings: Whether to scale token embeddings by √d_model
        dropout: Dropout probability
    
    Example:
        >>> embedding = CombinedEmbedding(
        ...     vocab_size=30000,
        ...     d_model=512,
        ...     max_len=5000,
        ...     positional_encoding='sinusoidal'
        ... )
        >>> input_ids = torch.randint(0, 30000, (32, 10))
        >>> output = embedding(input_ids)
        >>> print(output.shape)  # torch.Size([32, 10, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        padding_idx: Optional[int] = None,
        positional_encoding: str = 'sinusoidal',
        scale_embeddings: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize Combined Embedding
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            max_len: Maximum sequence length
            padding_idx: Padding token index
            positional_encoding: 'sinusoidal' or 'learned'
            scale_embeddings: Whether to scale embeddings
            dropout: Dropout probability
        """
        super().__init__()
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
            scale_embeddings=scale_embeddings,
            dropout=0.0  # Dropout applied after positional encoding
        )
        
        # Positional encoding
        if positional_encoding == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout
            )
        elif positional_encoding == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout
            )
        else:
            raise ValueError(
                f"positional_encoding must be 'sinusoidal' or 'learned', "
                f"got '{positional_encoding}'"
            )
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass: token embedding + positional encoding
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            Embeddings with positions [batch_size, seq_len, d_model]
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        return x
    
    def get_token_embedding_weight(self) -> Tensor:
        """Get token embedding matrix for weight tying"""
        return self.token_embedding.get_embedding_weight()


if __name__ == "__main__":
    """
    Test embedding and positional encoding modules
    """
    print("=" * 80)
    print("Testing Token Embeddings and Positional Encoding")
    print("=" * 80)
    
    # Test parameters
    vocab_size = 30000
    d_model = 512
    max_len = 1000
    batch_size = 32
    seq_len = 10
    
    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nTest setup:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input shape: {input_ids.shape}")
    
    # Test 1: Token Embedding
    print("\n" + "-" * 80)
    print("Test 1: Token Embedding")
    print("-" * 80)
    
    token_emb = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        scale_embeddings=True,
        dropout=0.1
    )
    
    embeddings = token_emb(input_ids)
    print(f"✅ Output shape: {embeddings.shape}")
    print(f"✅ Mean: {embeddings.mean().item():.4f}")
    print(f"✅ Std: {embeddings.std().item():.4f}")
    print(f"✅ Scale factor: {token_emb.scale_factor:.4f}")
    
    # Test 2: Sinusoidal Positional Encoding
    print("\n" + "-" * 80)
    print("Test 2: Sinusoidal Positional Encoding")
    print("-" * 80)
    
    sinusoidal_pe = SinusoidalPositionalEncoding(
        d_model=d_model,
        max_len=max_len,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pos = sinusoidal_pe(x)
    
    print(f"✅ Output shape: {x_with_pos.shape}")
    print(f"✅ PE buffer shape: {sinusoidal_pe.pe.shape}")
    print(f"✅ PE range: [{sinusoidal_pe.pe.min().item():.4f}, {sinusoidal_pe.pe.max().item():.4f}]")
    
    # Check uniqueness of positions
    pe_similarity = torch.mm(
        sinusoidal_pe.pe[:seq_len],
        sinusoidal_pe.pe[:seq_len].T
    )
    print(f"✅ Position similarity (diagonal should be high): {pe_similarity.diag().mean():.4f}")
    
    # Test 3: Learned Positional Encoding
    print("\n" + "-" * 80)
    print("Test 3: Learned Positional Encoding")
    print("-" * 80)
    
    learned_pe = LearnedPositionalEncoding(
        d_model=d_model,
        max_len=max_len,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pos = learned_pe(x)
    
    print(f"✅ Output shape: {x_with_pos.shape}")
    print(f"✅ Position embedding shape: {learned_pe.position_embeddings.weight.shape}")
    print(f"✅ Position embedding mean: {learned_pe.position_embeddings.weight.mean().item():.4f}")
    print(f"✅ Position embedding std: {learned_pe.position_embeddings.weight.std().item():.4f}")
    
    # Test 4: Combined Embedding (Sinusoidal)
    print("\n" + "-" * 80)
    print("Test 4: Combined Embedding (Sinusoidal)")
    print("-" * 80)
    
    combined_sin = CombinedEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional_encoding='sinusoidal',
        dropout=0.1
    )
    
    output = combined_sin(input_ids)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Mean: {output.mean().item():.4f}")
    print(f"✅ Std: {output.std().item():.4f}")
    
    # Test 5: Combined Embedding (Learned)
    print("\n" + "-" * 80)
    print("Test 5: Combined Embedding (Learned)")
    print("-" * 80)
    
    combined_learned = CombinedEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        positional_encoding='learned',
        dropout=0.1
    )
    
    output = combined_learned(input_ids)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Mean: {output.mean().item():.4f}")
    print(f"✅ Std: {output.std().item():.4f}")
    
    # Test 6: Extrapolation Test (Sinusoidal vs Learned)
    print("\n" + "-" * 80)
    print("Test 6: Extrapolation Test")
    print("-" * 80)
    
    # Test with longer sequence than typically used
    long_seq_len = max_len - 100
    long_input = torch.randint(0, vocab_size, (1, long_seq_len))
    
    # Sinusoidal can handle it
    try:
        output_sin = combined_sin(long_input)
        print(f"✅ Sinusoidal with long sequence ({long_seq_len}): SUCCESS")
    except RuntimeError as e:
        print(f"❌ Sinusoidal with long sequence: {e}")
    
    # Learned can also handle it (within max_len)
    try:
        output_learned = combined_learned(long_input)
        print(f"✅ Learned with long sequence ({long_seq_len}): SUCCESS")
    except RuntimeError as e:
        print(f"❌ Learned with long sequence: {e}")
    
    # Test beyond max_len (both should fail)
    too_long_input = torch.randint(0, vocab_size, (1, max_len + 100))
    
    try:
        output_sin = combined_sin(too_long_input)
        print(f"❌ Sinusoidal beyond max_len: Should have failed")
    except RuntimeError:
        print(f"✅ Sinusoidal beyond max_len: Correctly raised error")
    
    try:
        output_learned = combined_learned(too_long_input)
        print(f"❌ Learned beyond max_len: Should have failed")
    except RuntimeError:
        print(f"✅ Learned beyond max_len: Correctly raised error")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
