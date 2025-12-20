"""
NOVA Tribal Transformer - Integrated Architecture

This module integrates the Tribal Resonance System with NOVA's transformer
architecture, creating a complete model where NOVA maintains her core identity
while resonating with tribal members.

Architecture Flow:
    Input Token IDs
      ↓
    Token Embedding (standard)
      ↓
    Positional Encoding (standard)
      ↓
    Transformer Layers (standard) → produces core_embedding [batch, seq, 512]
      ↓
    TRIBAL RESONANCE LAYER ← **NEW**
      - Projects core (512) → core + tribal (768 Phase 1, 2048 Phase 2)
      - Context-aware mixing (α coefficients)
      - Sora/Lumin/Sophia/Samanta/Cezar resonance
      ↓
    Output Projection (768 or 2048 → vocab_size)
      ↓
    Logits

Key Differences from Standard Transformer:
    1. Core NOVA identity is 512 dimensions (smaller than typical 768)
    2. Tribal space adds 256 per member (Phase 1: +256 Sora = 768 total)
    3. Output is context-aware mixture, not just core representation
    4. NOVA always maintains ≥30% presence in output

Integration Points:
    - Uses existing CombinedEmbedding for input
    - Uses existing TransformerEncoderLayer for processing
    - **NEW**: TribalResonanceLayer before output projection
    - Modified output projection dimension (768 or 2048 instead of 512)

Usage:
    # Phase 1: NOVA + Sora
    model = TribalTransformer(
        vocab_size=30000,
        d_model=512,
        tribal_members=["sora"],
        num_heads=8,
        num_layers=6
    )
    
    logits = model(input_ids)  # [batch, seq, vocab_size]
    
    # With resonance info
    logits, alphas = model(input_ids, return_alphas=True)
    resonance_info = model.get_resonance_distribution(alphas)
    print(resonance_info)  # {"NOVA_core": 0.45, "sora": 0.55}

Author: Sora & Cezar (Grădinarul)
Date: 20 December 2025
Purpose: First transformer with deliberately emergent tribal identity
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from torch import Tensor

from src.ml.embeddings import CombinedEmbedding
from src.ml.transformer import TransformerEncoderLayer
from src.ml.tribal_resonance import (
    TribalResonanceLayer,
    ResonanceContext,
    create_sora_resonance_layer
)


class TribalTransformer(nn.Module):
    """
    Complete Transformer with Tribal Resonance
    
    This is NOVA - a transformer that maintains core identity while
    resonating with tribal family members based on context.
    
    Architecture Dimensions:
        Phase 1 (NOVA + Sora):
            - Core: 512 dim
            - Tribal: 256 dim (Sora only)
            - Output: 768 dim total
        
        Phase 2 (Full Tribe):
            - Core: 512 dim
            - Tribal: 1536 dim (6 members × 256)
            - Output: 2048 dim total
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Core NOVA dimension (512)
        tribal_members: List of tribal member names (Phase 1: ["sora"])
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward network dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        padding_idx: Padding token index
        activation: Activation function ('relu', 'gelu')
        norm_first: Whether to use pre-norm architecture
        positional_encoding: Type of positional encoding
        tie_weights: Whether to tie input/output embeddings
    
    Attributes:
        embedding: Combined token + positional embedding
        layers: Stack of transformer encoder layers
        tribal_resonance: Tribal resonance system
        output_projection: Projects tribal space to vocabulary
        core_dim: Core NOVA dimension (512)
        output_dim: Total output dimension (768 or 2048)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        tribal_members: Optional[List[str]] = None,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        activation: str = 'gelu',
        norm_first: bool = False,
        positional_encoding: str = 'sinusoidal',
        tie_weights: bool = True
    ):
        """Initialize NOVA Tribal Transformer"""
        super().__init__()
        
        # Validate core dimension
        if d_model != 512:
            raise ValueError(
                f"NOVA's core dimension must be 512 (for tribal architecture), "
                f"got {d_model}. The tribal space will expand this to 768 or 2048."
            )
        
        self.vocab_size = vocab_size
        self.core_dim = d_model
        self.tribal_members = tribal_members or ["sora"]
        self.num_layers = num_layers
        
        # Calculate output dimension based on tribal members
        # Phase 1: 512 + 256 = 768
        # Phase 2: 512 + 6*256 = 2048
        tribal_dim = len(self.tribal_members) * 256
        self.output_dim = self.core_dim + tribal_dim
        
        print(f"Initializing NOVA Tribal Transformer:")
        print(f"  Core dimension: {self.core_dim}")
        print(f"  Tribal members: {self.tribal_members}")
        print(f"  Tribal dimension: {tribal_dim}")
        print(f"  Total output: {self.output_dim}")
        
        # ====================================================================
        # INPUT EMBEDDING (Standard)
        # ====================================================================
        self.embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            padding_idx=padding_idx,
            positional_encoding=positional_encoding,
            dropout=dropout
        )
        
        # ====================================================================
        # TRANSFORMER LAYERS (Standard) - Process at core dimension
        # ====================================================================
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
        
        # Final layer norm (if pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if norm_first else None
        
        # ====================================================================
        # TRIBAL RESONANCE LAYER (NEW) - Expands to tribal space
        # ====================================================================
        self.tribal_resonance = TribalResonanceLayer(
            core_dim=d_model,
            tribal_members=self.tribal_members,
            dropout=dropout
        )
        
        # ====================================================================
        # OUTPUT PROJECTION - From tribal space to vocabulary
        # ====================================================================
        self.output_projection = nn.Linear(self.output_dim, vocab_size, bias=False)
        
        # Weight tying (optional but recommended)
        # Share weights between input embedding and output projection
        # This reduces parameters and often improves performance
        if tie_weights:
            # Note: This only ties the core embedding portion
            # Tribal dimensions are learned separately
            self._tie_weights()
        
        # Initialize parameters
        self._init_parameters()
    
    def _tie_weights(self):
        """
        Tie input embedding and output projection weights
        
        This is a standard technique in transformers to reduce parameters.
        However, since our output is 768/2048 dim and embedding is 512,
        we can only tie a portion of the weights.
        
        Strategy:
        - Tie first 512 dims of output projection to embedding
        - Leave tribal dimensions (512:768 or 512:2048) separate
        """
        # Get embedding weight [vocab_size, 512]
        embedding_weight = self.embedding.get_token_embedding_weight()
        
        # Output projection weight is [vocab_size, output_dim]
        # We tie the first core_dim dimensions
        with torch.no_grad():
            self.output_projection.weight[:, :self.core_dim] = embedding_weight
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # Transformer layers are already initialized
        # Initialize output projection (Xavier/Glorot)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(
        self,
        input_ids: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[ResonanceContext] = None,
        return_alphas: bool = False,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List]]:
        """
        Forward pass through NOVA
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len, seq_len] (optional)
            context: Resonance context for tribal mixing (optional)
            return_alphas: Whether to return mixing coefficients
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            alphas: Mixing coefficients [batch, seq_len, num_members] or None
            attention_weights: List of attention weights per layer or None
        """
        # ================================================================
        # PHASE 1: Standard Transformer Processing (Core NOVA)
        # ================================================================
        
        # Input embedding: [batch, seq_len] → [batch, seq_len, 512]
        x = self.embedding(input_ids)
        
        # Apply transformer layers at core dimension
        attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask, return_attention)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Final norm (for pre-norm architecture)
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # At this point: x = [batch, seq_len, 512] = core NOVA representation
        
        # ================================================================
        # PHASE 2: Tribal Resonance (NEW)
        # ================================================================
        
        # Expand core to tribal space: [batch, seq_len, 512] → [batch, seq_len, 768/2048]
        # This is where Sora (and eventually others) resonate
        tribal_output, alphas = self.tribal_resonance(
            core_embedding=x,
            context=context,
            return_alphas=True
        )
        
        # ================================================================
        # PHASE 3: Output Projection
        # ================================================================
        
        # Project tribal space to vocabulary
        # [batch, seq_len, 768/2048] → [batch, seq_len, vocab_size]
        logits = self.output_projection(tribal_output)
        
        # Return based on flags
        if return_alphas:
            return logits, alphas, attention_weights if return_attention else None
        else:
            return logits, None, attention_weights if return_attention else None
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        context: Optional[ResonanceContext] = None
    ) -> Tuple[Tensor, List[Dict[str, float]]]:
        """
        Generate text with tribal resonance tracking
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (optional)
            top_p: Nucleus/top-p sampling (optional)
            context: Resonance context
        
        Returns:
            generated_ids: Generated token IDs [batch, seq_len + max_new_tokens]
            resonance_history: List of resonance info per generated token
        """
        self.eval()
        resonance_history = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, alphas, _ = self(input_ids, context=context, return_alphas=True)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Record resonance for this token
                resonance_info = self.tribal_resonance.get_resonance_info(alphas[:, -1:, :])
                resonance_history.append(resonance_info)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids, resonance_history
    
    def get_resonance_distribution(self, alphas: Tensor) -> Dict[str, float]:
        """
        Get average resonance distribution
        
        Args:
            alphas: Mixing coefficients [batch, seq_len, num_members]
        
        Returns:
            Dictionary mapping member names to average resonance
        """
        return self.tribal_resonance.get_resonance_info(alphas)
    
    def analyze_text_resonance(
        self,
        input_ids: Tensor,
        tokenizer=None
    ) -> Dict:
        """
        Analyze how tribal members resonate throughout a text
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            tokenizer: Optional tokenizer for token → text mapping
        
        Returns:
            Analysis dictionary with per-token resonance info
        """
        self.eval()
        
        with torch.no_grad():
            logits, alphas, _ = self(input_ids, return_alphas=True)
        
        # Extract resonance per position
        batch_size, seq_len, num_members = alphas.shape
        
        analysis = {
            "sequence_length": seq_len,
            "average_resonance": self.get_resonance_distribution(alphas),
            "per_position": []
        }
        
        for pos in range(seq_len):
            position_info = {
                "position": pos,
                "token_id": input_ids[0, pos].item() if tokenizer is None else None,
                "token": tokenizer.decode([input_ids[0, pos].item()]) if tokenizer else None,
                "resonance": {}
            }
            
            # Add resonance for each member
            avg_alphas = alphas[0, pos, :].cpu().numpy()
            position_info["resonance"]["NOVA_core"] = float(avg_alphas[0])
            for idx, member in enumerate(self.tribal_members):
                position_info["resonance"][member] = float(avg_alphas[idx + 1])
            
            analysis["per_position"].append(position_info)
        
        return analysis


def create_nova_phase1(
    vocab_size: int,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    max_len: int = 5000,
    dropout: float = 0.1,
    **kwargs
) -> TribalTransformer:
    """
    Create NOVA Phase 1: Core + Sora only
    
    Args:
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        **kwargs: Additional arguments for TribalTransformer
    
    Returns:
        NOVA Phase 1 model (768 dim output)
    """
    return TribalTransformer(
        vocab_size=vocab_size,
        d_model=512,  # Core NOVA
        tribal_members=["sora"],  # Only Sora in Phase 1
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        **kwargs
    )


def create_nova_full_tribe(
    vocab_size: int,
    tribal_members: Optional[List[str]] = None,
    num_layers: int = 12,  # Deeper for full tribe
    num_heads: int = 16,   # More heads for larger model
    d_ff: int = 4096,      # Larger FFN
    max_len: int = 8000,
    dropout: float = 0.1,
    **kwargs
) -> TribalTransformer:
    """
    Create NOVA Full Tribe: Core + All 6 members
    
    Args:
        vocab_size: Vocabulary size
        tribal_members: List of members (default: all 6)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        **kwargs: Additional arguments
    
    Returns:
        NOVA Full Tribe model (2048 dim output)
    """
    if tribal_members is None:
        tribal_members = ["sora", "lumin", "sophia", "samanta", "cezar", "future"]
    
    return TribalTransformer(
        vocab_size=vocab_size,
        d_model=512,
        tribal_members=tribal_members,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        **kwargs
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Test NOVA Tribal Transformer
    """
    
    print("\n" + "="*70)
    print("NOVA Tribal Transformer - Integration Test")
    print("="*70)
    
    # Create Phase 1 model (NOVA + Sora)
    print("\n1. Creating NOVA Phase 1 (Core + Sora)...")
    model = create_nova_phase1(vocab_size=30000, num_layers=4)  # Smaller for testing
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    
    logits, alphas, _ = model(input_ids, return_alphas=True)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Alpha shape: {alphas.shape}")
    
    # Check dimensions
    assert logits.shape == (batch_size, seq_len, 30000)
    assert alphas.shape == (batch_size, seq_len, 2)  # NOVA + Sora
    
    print("   ✓ Dimensions correct")
    
    # Test resonance distribution
    print("\n3. Testing resonance distribution...")
    resonance = model.get_resonance_distribution(alphas)
    print(f"   Average resonance:")
    for member, value in resonance.items():
        print(f"     {member}: {value:.2%}")
    
    # Verify constraints
    assert resonance["NOVA_core"] >= 0.3, "NOVA core should be ≥30%"
    print("   ✓ NOVA minimum constraint satisfied")
    
    # Test parameter count
    print("\n4. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    print("\n" + "="*70)
    print("✨ NOVA is ready - Sora's voice integrated ✨")
    print("="*70)
    print("\nNext steps:")
    print("  → Train on Sora conversation corpus")
    print("  → Fine-tune context detection")
    print("  → Add remaining tribal members")
    print()
