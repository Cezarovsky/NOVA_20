# NOVA AI System - Implementation Roadmap

**Project**: NOVA - Multi-Context AI Assistant
**Version**: 1.0
**Date**: 28 Noiembrie 2025
**Status**: Planning & Implementation Phase

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Development Phases](#2-development-phases)
3. [Phase 1: Foundation & Core Infrastructure](#phase-1-foundation--core-infrastructure)
4. [Phase 2: Specialized Agents Implementation](#phase-2-specialized-agents-implementation)
5. [Phase 3: Orchestrator & Integration](#phase-3-orchestrator--integration)
6. [Phase 4: User Interface & Experience](#phase-4-user-interface--experience)
7. [Phase 5: Testing & Optimization](#phase-5-testing--optimization)
8. [Phase 6: Advanced Features](#phase-6-advanced-features)
9. [Technical Stack Summary](#technical-stack-summary)
10. [Success Metrics](#success-metrics)

---

## 1. PROJECT OVERVIEW

### 1.1 Mission Statement
Crearea unui sistem AI modular care procesează și sintetizează informații din multiple surse (documente, imagini, audio, web) folosind arhitectură multi-agent și tehnologii open-source/privacy-focused (fără OpenAI).

### 1.2 Core Principles
- ✅ **Modularitate**: Componente independente și reutilizabile
- ✅ **Privacy-First**: Anthropic Claude + Mistral + modele locale (no OpenAI)
- ✅ **Scalabilitate**: Arhitectură pregătită pentru growth
- ✅ **Explicabilitate**: Citare surse și transparență în procesare
- ✅ **Performance**: Optimizări pentru latență și cost

### 1.3 Technology Stack
- **LLMs**: Anthropic Claude 3.5 Sonnet, Mistral Large/Small
- **Embeddings**: mistral-embed (1024D)
- **Vision**: Claude 3.5 Sonnet Vision
- **Audio**: Faster-Whisper (local, open-source)
- **Vector DB**: ChromaDB (local-first)
- **UI**: Streamlit
- **Language**: Python 3.11+

---

## 2. DEVELOPMENT PHASES

### Phase Overview

| Phase | Focus | Duration | Priority | Status |
|-------|-------|----------|----------|--------|
| **Phase 1** | Foundation & Core ML Infrastructure | 2 weeks | Critical | 🔄 In Progress |
| **Phase 2** | Transformer Implementation & Inference | 2 weeks | Critical | ⏳ Pending |
| **Phase 3** | Optimization & Caching (KV Cache, Top-K) | 1 week | Critical | ⏳ Pending |
| **Phase 4** | Basic Agent & Orchestrator | 1 week | High | ⏳ Pending |
| **Phase 5** | UI & Testing | 1 week | High | ⏳ Pending |
| **Phase 6** | Advanced Features (Optional) | 2 weeks | Low | ⏳ Pending |

**Total Estimated Timeline**: 7 weeks for MVP with strong ML foundations

**Focus**: Deep Learning fundamentals first, specialized agents secondary

---

## PHASE 1: Foundation & Core ML Infrastructure

**Goal**: Implementarea bazelor de deep learning și infrastructură pentru Transformer models

**Duration**: 2 săptămâni (10-14 zile)

**Status**: 🔄 In Progress (15% complete)

**Focus Areas**:
1. Core Transformer components (Attention, Feed-Forward)
2. Token embeddings & positional encoding
3. Inference optimization (KV cache, top-k sampling)
4. Basic LLM interface with API providers

### 1.1 Project Structure Setup ✅ DONE

**Status**: ✅ Complete

**Tasks**:
- [x] Create project directory structure
- [x] Setup virtual environment
- [x] Create requirements.txt
- [x] Install all dependencies
- [x] Create Documentation folder

**Output**:
```
Nova_20/
├── Documentation/
│   ├── arhitectura_nova.md
│   └── implementation.md
├── venv/
└── requirements.txt
```

### 1.2 Core Directory Structure 🔄 CURRENT

**Status**: 🔄 In Progress

**Tasks**:
- [ ] Create source code directory structure
- [ ] Setup configuration management
- [ ] Create .env.example template
- [ ] Setup logging configuration
- [ ] Create __init__.py files

**Expected Structure**:
```
Nova_20/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration management
│   │   └── logging_config.py    # Logging setup
│   ├── ml/                       # Machine Learning Core (NEW - Priority)
│   │   ├── __init__.py
│   │   ├── attention.py          # Attention mechanisms (Multi-Head, Scaled Dot-Product)
│   │   ├── transformer.py        # Transformer blocks & architecture
│   │   ├── embeddings.py         # Token & positional embeddings
│   │   ├── inference.py          # Inference engine with KV cache
│   │   ├── sampling.py           # Top-k, top-p, temperature sampling
│   │   └── optimization.py       # Inference optimizations
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm_interface.py     # API-based LLM interface (Anthropic, Mistral)
│   │   └── vector_store.py      # ChromaDB wrapper
│   ├── agents/                   # Lower priority
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   └── orchestrator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tokenizers.py        # Tokenization utilities
│   │   └── metrics.py           # Performance metrics
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   ├── test_ml/                  # ML components tests
│   ├── test_core/
│   └── test_inference/           # Inference optimization tests
├── data/
│   ├── chroma_db/
│   └── cache/                    # KV cache storage
├── logs/
├── Documentation/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

**Implementation Steps**:
1. Create all directories
2. Create all __init__.py files
3. Setup .gitignore
4. Create .env.example with required API keys
5. Create basic README.md

**Acceptance Criteria**:
- ✅ All directories created
- ✅ Import structure works (no circular dependencies)
- ✅ .gitignore configured properly
- ✅ .env.example documented

### 1.3 Transformer Attention Mechanism ⚡ PRIORITY

**Status**: ⏳ Pending

**File**: `src/ml/attention.py`

**Priority**: CRITICAL (Core ML Foundation)

**Tasks**:
- [ ] Implement Scaled Dot-Product Attention
- [ ] Implement Multi-Head Attention
- [ ] Add attention masking (causal for autoregressive)
- [ ] Implement attention visualization utilities
- [ ] Add numerical stability improvements

**Mathematical Foundation**:

**Scaled Dot-Product Attention**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q$ (Query): $(batch, seq\_len, d_k)$
- $K$ (Key): $(batch, seq\_len, d_k)$
- $V$ (Value): $(batch, seq\_len, d_v)$
- $d_k$: dimension of keys/queries (scaling factor prevents softmax saturation)

**Implementation**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,     # (batch, n_heads, seq_len, d_k)
        key: torch.Tensor,       # (batch, n_heads, seq_len, d_k)
        value: torch.Tensor,     # (batch, n_heads, seq_len, d_v)
        mask: Optional[torch.Tensor] = None,  # (batch, 1, seq_len, seq_len) or (seq_len, seq_len)
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask (1 = attend, 0 = mask out)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attention output (batch, n_heads, seq_len, d_v)
            attention_weights: Optional attention matrix
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # (batch, n_heads, seq_len_q, d_k) @ (batch, n_heads, d_k, seq_len_k)
        # -> (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (set masked positions to large negative value before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len_q, seq_len_k)
        
        # Apply dropout (during training)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)  # (batch, n_heads, seq_len, d_v)
        
        if return_attention:
            return output, attention_weights
        return output, None


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    
    Allows model to jointly attend to information from different
    representation subspaces at different positions.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(
        self,
        d_model: int,        # Model dimension (e.g., 768, 1024)
        n_heads: int,        # Number of attention heads (e.g., 8, 12)
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dimension into (n_heads, d_k)
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads
        
        Args:
            x: (batch, n_heads, seq_len, d_k)
        
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k)
        x = x.transpose(1, 2)
        # (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,     # (batch, seq_len, d_model)
        key: torch.Tensor,       # (batch, seq_len, d_model)
        value: torch.Tensor,     # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            return_attention: Return attention weights
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: Optional (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)    # (batch, seq_len, d_model)
        V = self.W_v(value)  # (batch, seq_len, d_model)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch, n_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch, n_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch, n_heads, seq_len, d_k)
        
        # 3. Apply attention
        attn_output, attention_weights = self.attention(
            Q, K, V, mask, return_attention
        )  # (batch, n_heads, seq_len, d_k)
        
        # 4. Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch, seq_len, d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)  # (batch, seq_len, d_model)
        output = self.dropout(output)
        
        return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive models
    
    Prevents attending to future tokens during training/generation
    
    Args:
        seq_len: Sequence length
        device: torch device
    
    Returns:
        mask: (seq_len, seq_len) with 1s in lower triangle, 0s above
    
    Example:
        seq_len = 4
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Create padding mask to ignore pad tokens
    
    Args:
        seq: Input sequence (batch, seq_len)
        pad_idx: Padding token index
    
    Returns:
        mask: (batch, 1, 1, seq_len)
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
```

**Key Concepts**:

1. **Why Scaled?** 
   - Division by $\sqrt{d_k}$ prevents dot products from growing too large
   - Large dot products → extreme softmax values → vanishing gradients

2. **Multi-Head Attention Benefits**:
   - Different heads learn different patterns
   - Some heads focus on local context, others on long-range dependencies
   - Increases model expressiveness

3. **Causal Masking**:
   - Essential for autoregressive models (GPT-style)
   - Prevents "cheating" by looking at future tokens
   - Lower triangular mask

**Acceptance Criteria**:
- ✅ Attention computation correct (verified with known examples)
- ✅ Multi-head splitting/combining works
- ✅ Causal mask prevents future attention
- ✅ Numerical stability (no NaN/Inf)
- ✅ Gradient flow verified
- ✅ Performance: < 10ms for seq_len=512, d_model=768

**Testing**:
```python
# test_attention.py
def test_attention_output_shape():
    batch, seq_len, d_model, n_heads = 2, 10, 768, 12
    attn = MultiHeadAttention(d_model, n_heads)
    
    x = torch.randn(batch, seq_len, d_model)
    output, _ = attn(x, x, x)
    
    assert output.shape == (batch, seq_len, d_model)

def test_causal_masking():
    # Verify future tokens are masked
    attn = ScaledDotProductAttention()
    Q = K = V = torch.randn(1, 4, 10, 64)
    mask = create_causal_mask(10, Q.device)
    
    _, weights = attn(Q, K, V, mask, return_attention=True)
    
    # Check upper triangle is zero (masked)
    upper_triangle = weights[0, 0].triu(diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)
```

### 1.4 KV Cache & Inference Optimization 🚀 PRIORITY

**Status**: ⏳ Pending

**File**: `src/ml/inference.py`

**Priority**: CRITICAL (Performance Foundation)

**Tasks**:
- [ ] Implement KV cache for autoregressive generation
- [ ] Implement efficient inference engine
- [ ] Add batch inference support
- [ ] Memory optimization for long sequences
- [ ] Performance profiling utilities

**Problem**: Why KV Cache?

In autoregressive generation, at each step $t$, we compute:
- New token embedding: $x_t$
- Attention over **all previous tokens**: $x_1, x_2, ..., x_t$

**Without KV Cache**:
```python
# Step 1: Generate token 1
Q1, K1, V1 = project(x1)
out1 = attention(Q1, K1, V1)  # Compute K1, V1

# Step 2: Generate token 2
Q2, K12, V12 = project(x1, x2)  # Recompute K1, V1 + compute K2, V2 ❌
out2 = attention(Q2, K12, V12)

# Step 3: Generate token 3
Q3, K123, V123 = project(x1, x2, x3)  # Recompute K1, K2, V1, V2 ❌❌
out3 = attention(Q3, K123, V123)
```

**Complexity**: $O(n^2)$ for generating $n$ tokens → wasteful recomputation!

**With KV Cache**:
```python
# Step 1
K_cache = [K1]
V_cache = [V1]
out1 = attention(Q1, K_cache, V_cache)

# Step 2
K_cache.append(K2)  # Only compute new K2
V_cache.append(V2)  # Only compute new V2
out2 = attention(Q2, K_cache, V_cache)

# Step 3
K_cache.append(K3)  # Only compute new K3
V_cache.append(V3)  # Only compute new V3
out3 = attention(Q3, K_cache, V_cache)
```

**Complexity**: $O(n)$ → **Linear time!** 🚀

**Speedup**: 10-100x faster for long sequences

**Implementation**:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation
    
    Stores past keys and values to avoid recomputation
    """
    keys: torch.Tensor     # (batch, n_heads, past_seq_len, d_k)
    values: torch.Tensor   # (batch, n_heads, past_seq_len, d_v)
    
    def update(
        self,
        new_keys: torch.Tensor,    # (batch, n_heads, 1, d_k)
        new_values: torch.Tensor   # (batch, n_heads, 1, d_v)
    ) -> 'KVCache':
        """
        Append new keys and values to cache
        
        Returns:
            Updated cache with concatenated keys/values
        """
        updated_keys = torch.cat([self.keys, new_keys], dim=2)
        updated_values = torch.cat([self.values, new_values], dim=2)
        return KVCache(keys=updated_keys, values=updated_values)
    
    @property
    def seq_len(self) -> int:
        """Return current sequence length in cache"""
        return self.keys.size(2)


class MultiHeadAttentionWithCache(nn.Module):
    """
    Multi-Head Attention with KV caching support
    
    Modified from standard MHA to support incremental generation
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,            # (batch, 1, d_model) for generation
        key: torch.Tensor,              # (batch, seq_len, d_model)
        value: torch.Tensor,            # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            query: Current query (typically 1 token during generation)
            key: Keys (full sequence if cache=None, else just new token)
            value: Values (full sequence if cache=None, else just new token)
            mask: Attention mask
            cache: Existing KV cache
            use_cache: Whether to return updated cache
        
        Returns:
            output: Attention output (batch, 1, d_model)
            new_cache: Updated KV cache if use_cache=True
        """
        batch_size = query.size(0)
        
        # 1. Project query (always for current token)
        Q = self.W_q(query)  # (batch, 1, d_model)
        Q = self._split_heads(Q)  # (batch, n_heads, 1, d_k)
        
        # 2. Handle keys and values with cache
        if cache is not None:
            # Incremental generation: only project new token
            K_new = self.W_k(key)  # (batch, 1, d_model)
            V_new = self.W_v(value)  # (batch, 1, d_model)
            K_new = self._split_heads(K_new)  # (batch, n_heads, 1, d_k)
            V_new = self._split_heads(V_new)  # (batch, n_heads, 1, d_v)
            
            # Update cache
            cache = cache.update(K_new, V_new)
            K = cache.keys    # (batch, n_heads, past_len+1, d_k)
            V = cache.values  # (batch, n_heads, past_len+1, d_v)
        else:
            # First pass or full sequence: project all
            K = self.W_k(key)    # (batch, seq_len, d_model)
            V = self.W_v(value)  # (batch, seq_len, d_model)
            K = self._split_heads(K)  # (batch, n_heads, seq_len, d_k)
            V = self._split_heads(V)  # (batch, n_heads, seq_len, d_v)
            
            if use_cache:
                cache = KVCache(keys=K, values=V)
        
        # 3. Compute attention
        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)  # (batch, n_heads, 1, d_v)
        
        # 4. Combine heads and project
        attn_output = self._combine_heads(attn_output)  # (batch, 1, d_model)
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        if use_cache:
            return output, cache
        return output, None
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)


class InferenceEngine:
    """
    Efficient inference engine with KV caching and optimizations
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        max_batch_size: int = 8
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.device = device
        self.max_batch_size = max_batch_size
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,      # (batch, input_seq_len)
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            use_cache: Use KV cache for speedup
            return_dict: Return additional info
        
        Returns:
            generated_ids: (batch, input_seq_len + max_new_tokens)
        """
        batch_size, input_len = input_ids.shape
        device = input_ids.device
        
        # Initialize
        generated_ids = input_ids.clone()
        cache_dict = {} if use_cache else None
        
        for step in range(max_new_tokens):
            # Prepare input for current step
            if use_cache and step > 0:
                # Only pass new token
                current_input = generated_ids[:, -1:]  # (batch, 1)
            else:
                # First step: pass full sequence
                current_input = generated_ids
            
            # Forward pass
            if use_cache:
                logits, cache_dict = self.model(
                    current_input,
                    cache=cache_dict,
                    use_cache=True
                )
            else:
                logits = self.model(current_input)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample next token
            next_token = self._sample_next_token(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )  # (batch, 1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token (if needed)
            # if (next_token == eos_token_id).all():
            #     break
        
        return generated_ids
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,     # (batch, vocab_size)
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample next token with top-k and/or top-p filtering
        
        Returns:
            (batch, 1)
        """
        # Apply top-k filtering
        if top_k is not None:
            logits = self._top_k_filtering(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            logits = self._top_p_filtering(logits, top_p)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)
        
        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        
        return next_token
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        Keep only top-k logits, set others to -inf
        
        Reduces diversity by only considering k most likely tokens
        """
        if top_k <= 0:
            return logits
        
        # Get top k values
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)  # (batch, top_k)
        min_values = values[:, -1:].unsqueeze(-1)  # (batch, 1, 1)
        
        # Mask out values below top-k threshold
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """
        Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p
        
        More dynamic than top-k: adjusts cutoff based on distribution
        
        Example:
            top_p = 0.9
            If top 3 tokens have probs [0.6, 0.3, 0.05, ...], keep only first 2
        """
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits


# Memory optimization utilities

def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    dtype: torch.dtype = torch.float32
) -> int:
    """
    Estimate KV cache memory requirement
    
    Returns:
        Memory in bytes
    """
    bytes_per_element = torch.finfo(dtype).bits // 8
    
    # Each layer stores K and V
    # Shape per layer: (batch, n_heads, seq_len, d_k) for K and V
    elements_per_layer = 2 * batch_size * n_heads * seq_len * d_k
    total_elements = elements_per_layer * n_layers
    
    memory_bytes = total_elements * bytes_per_element
    
    return memory_bytes


def estimate_kv_cache_memory_human(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    dtype: torch.dtype = torch.float32
) -> str:
    """
    Human-readable memory estimate
    
    Example:
        "KV Cache Memory: 2.4 GB"
    """
    memory_bytes = estimate_kv_cache_memory(
        batch_size, seq_len, n_layers, n_heads, d_k, dtype
    )
    
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 ** 3:
        return f"{memory_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_bytes / (1024 ** 3):.2f} GB"
```

**Key Optimizations**:

1. **KV Cache**: Avoid recomputing past keys/values
   - **Speedup**: 10-100x for long sequences
   - **Memory**: $O(n)$ where $n$ = sequence length

2. **Top-k Sampling**: Only consider k most likely tokens
   - Faster sampling (smaller search space)
   - Controls diversity

3. **Top-p (Nucleus) Sampling**: Dynamic cutoff
   - Adapts to distribution shape
   - Better quality than top-k

4. **Memory Estimation**: Plan ahead for large models
   - Example: GPT-3 scale (96 layers, 96 heads, 128 d_k), seq_len=2048, batch=1
   - KV Cache: **~18 GB** for float32!

**Performance Comparison**:

| Method | Tokens/sec | Memory | Quality |
|--------|------------|--------|---------|
| Naive (no cache) | 10-20 | Low | Same |
| With KV Cache | 100-500 | High | Same |
| + Top-k (k=50) | 200-800 | High | Good |
| + Top-p (p=0.9) | 150-600 | High | Better |

**Acceptance Criteria**:
- ✅ KV cache correctly stores/retrieves K, V
- ✅ Generation is 10x+ faster with cache
- ✅ Top-k filtering works correctly
- ✅ Top-p (nucleus) sampling works
- ✅ Memory estimation accurate
- ✅ No memory leaks during long generation

**Testing**:
```python
# test_inference.py
def test_kv_cache_speedup():
    # Compare generation speed with/without cache
    model = ...
    engine = InferenceEngine(model)
    
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Without cache
    start = time.time()
    _ = engine.generate(input_ids, max_new_tokens=100, use_cache=False)
    time_no_cache = time.time() - start
    
    # With cache
    start = time.time()
    _ = engine.generate(input_ids, max_new_tokens=100, use_cache=True)
    time_with_cache = time.time() - start
    
    speedup = time_no_cache / time_with_cache
    assert speedup > 5, f"Expected 5x+ speedup, got {speedup:.2f}x"

def test_top_k_sampling():
    logits = torch.randn(1, 1000)
    engine = InferenceEngine(None)
    
    # Top-k should zero out all but top k
    filtered = engine._top_k_filtering(logits, top_k=10)
    non_inf = (filtered != float('-inf')).sum()
    
    assert non_inf == 10, f"Expected 10 non-inf values, got {non_inf}"
```

### 1.5 Top-K Sampling Strategies 🎲 PRIORITY

**Status**: ⏳ Pending

**File**: `src/ml/sampling.py`

**Priority**: HIGH (Generation Quality)

**Tasks**:
- [ ] Implement various sampling strategies
- [ ] Temperature scaling
- [ ] Top-k filtering
- [ ] Top-p (nucleus) sampling
- [ ] Beam search
- [ ] Greedy decoding

**Sampling Theory**:

**Problem**: Given logits from model, how do we choose next token?

**Strategies**:

#### 1. Greedy Decoding (Deterministic)
```python
# Always pick most probable token
next_token = torch.argmax(logits, dim=-1)
```
- **Pros**: Fast, deterministic
- **Cons**: Repetitive, boring output

#### 2. Temperature Sampling
```python
# Scale logits before softmax
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```
- `temperature = 0.7`: More focused (sharper distribution)
- `temperature = 1.0`: Normal
- `temperature = 1.5`: More random (flatter distribution)

**Mathematical Effect**:
$$P(x_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Lower $T$ → higher probability for top tokens

#### 3. Top-K Sampling
```python
# Keep only top k most probable tokens
top_k_logits = filter_top_k(logits, k=50)
probs = F.softmax(top_k_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```
- Reduces noise from low-probability tokens
- Fixed cutoff (k tokens)

#### 4. Top-P (Nucleus) Sampling
```python
# Keep smallest set with cumulative prob >= p
top_p_logits = filter_top_p(logits, p=0.9)
probs = F.softmax(top_p_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```
- Dynamic cutoff (adapts to distribution)
- More sophisticated than top-k

**Full Implementation**:

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from enum import Enum

class SamplingStrategy(Enum):
    """Available sampling strategies"""
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"


class TextSampler:
    """
    Advanced text sampling with multiple strategies
    """
    
    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.TOP_P,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0
    ):
        self.strategy = strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
    
    def sample(
        self,
        logits: torch.Tensor,          # (batch, vocab_size)
        generated_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample next token(s) using configured strategy
        
        Args:
            logits: Raw logits from model
            generated_tokens: Previously generated tokens (for repetition penalty)
        
        Returns:
            next_token_ids: (batch, 1)
        """
        # Apply repetition penalty
        if generated_tokens is not None and self.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, generated_tokens, self.repetition_penalty
            )
        
        # Route to appropriate sampling method
        if self.strategy == SamplingStrategy.GREEDY:
            return self._greedy_sample(logits)
        elif self.strategy == SamplingStrategy.TEMPERATURE:
            return self._temperature_sample(logits, self.temperature)
        elif self.strategy == SamplingStrategy.TOP_K:
            return self._top_k_sample(logits, self.top_k, self.temperature)
        elif self.strategy == SamplingStrategy.TOP_P:
            return self._top_p_sample(logits, self.top_p, self.temperature)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding: always pick most probable token
        
        Fast but often repetitive
        """
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    def _temperature_sample(
        self,
        logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Sample with temperature scaling
        
        Args:
            temperature: 
                < 1.0: More confident (peaked distribution)
                = 1.0: Normal
                > 1.0: More random (flat distribution)
        """
        if temperature == 0:
            return self._greedy_sample(logits)
        
        # Scale logits
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample from distribution
        return torch.multinomial(probs, num_samples=1)
    
    def _top_k_sample(
        self,
        logits: torch.Tensor,
        top_k: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Top-K sampling: only consider k most probable tokens
        
        Reduces tail noise while maintaining diversity
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Get top k values and indices
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k, dim=-1)
        
        # Create tensor with -inf for all positions
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        # Scatter top-k values back
        filtered_logits.scatter_(-1, indices, values)
        
        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def _top_p_sample(
        self,
        logits: torch.Tensor,
        top_p: float,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Top-P (Nucleus) sampling: dynamic cutoff based on cumulative probability
        
        More sophisticated than top-k: adapts to distribution shape
        
        Example:
            If top tokens have probs [0.5, 0.3, 0.1, 0.05, ...] and top_p=0.9,
            keep first 3 tokens (0.5 + 0.3 + 0.1 = 0.9)
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        
        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff: first position where cumsum > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep at least one token (shift right)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        # Mask out removed indices
        filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to reduce token repetition
        
        Reduces logits for previously generated tokens
        
        Args:
            penalty: 
                > 1.0: Penalize repetition (typical: 1.1-1.5)
                = 1.0: No penalty
                < 1.0: Encourage repetition (unusual)
        """
        if penalty == 1.0:
            return logits
        
        # For each previously generated token, divide its logit by penalty
        for batch_idx in range(logits.size(0)):
            for token_id in generated_tokens[batch_idx].unique():
                # If logit is positive, divide; if negative, multiply
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty
        
        return logits


def compare_sampling_strategies():
    """
    Visualization: Compare different sampling strategies
    
    Shows how each strategy affects the probability distribution
    """
    import matplotlib.pyplot as plt
    
    # Create example logits (peaked distribution)
    logits = torch.tensor([
        [5.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1, -0.5, -1.0]
    ])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original distribution
    probs = F.softmax(logits, dim=-1)[0].numpy()
    axes[0, 0].bar(range(len(probs)), probs)
    axes[0, 0].set_title("Original Distribution")
    axes[0, 0].set_ylabel("Probability")
    
    # 2. Temperature = 0.5 (peaked)
    temp_logits = logits / 0.5
    temp_probs = F.softmax(temp_logits, dim=-1)[0].numpy()
    axes[0, 1].bar(range(len(temp_probs)), temp_probs)
    axes[0, 1].set_title("Temperature = 0.5")
    
    # 3. Temperature = 1.5 (flat)
    temp_logits = logits / 1.5
    temp_probs = F.softmax(temp_logits, dim=-1)[0].numpy()
    axes[0, 2].bar(range(len(temp_probs)), temp_probs)
    axes[0, 2].set_title("Temperature = 1.5")
    
    # 4. Top-k = 3
    sampler = TextSampler(strategy=SamplingStrategy.TOP_K, top_k=3)
    filtered = sampler._top_k_sample(logits, 3, 1.0)
    # Show which tokens are kept
    axes[1, 0].bar(range(len(probs)), probs)
    axes[1, 0].axvline(x=2.5, color='r', linestyle='--', label='Top-3 cutoff')
    axes[1, 0].set_title("Top-K = 3")
    axes[1, 0].legend()
    
    # 5. Top-p = 0.8
    # Highlight tokens in nucleus
    sorted_probs = torch.sort(torch.tensor(probs), descending=True)[0]
    cumsum = torch.cumsum(sorted_probs, dim=0)
    nucleus_size = (cumsum <= 0.8).sum().item() + 1
    axes[1, 1].bar(range(len(probs)), probs)
    axes[1, 1].set_title(f"Top-P = 0.8 (nucleus size: {nucleus_size})")
    
    # 6. Comparison
    axes[1, 2].text(
        0.1, 0.5,
        "Comparison:\n\n"
        "Greedy: Pick argmax\n"
        "Temperature: Scale randomness\n"
        "Top-K: Fixed cutoff\n"
        "Top-P: Dynamic cutoff\n\n"
        "Best: Top-P with temp=0.7-1.0",
        fontsize=12,
        verticalalignment='center'
    )
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("sampling_strategies.png", dpi=150)
    print("Saved comparison to sampling_strategies.png")
```

**Practical Recommendations**:

| Task | Strategy | Temperature | Top-K | Top-P |
|------|----------|-------------|-------|-------|
| Code Generation | Top-K | 0.2 | 5-10 | - |
| Creative Writing | Top-P | 0.8-1.0 | - | 0.9-0.95 |
| Q&A (Factual) | Top-K | 0.1-0.3 | 3-5 | - |
| Chatbot | Top-P | 0.7 | - | 0.9 |
| Translation | Greedy or Top-K | 0.1 | 1-3 | - |

**Acceptance Criteria**:
- ✅ All sampling strategies implemented
- ✅ Temperature scaling works correctly
- ✅ Top-k filtering correct
- ✅ Top-p (nucleus) sampling correct
- ✅ Repetition penalty reduces repetition
- ✅ Output quality verified manually

**Testing**:
```python
# test_sampling.py
def test_temperature_effect():
    sampler = TextSampler(strategy=SamplingStrategy.TEMPERATURE)
    logits = torch.tensor([[5.0, 2.0, 1.0, 0.5]])
    
    # Low temp should pick first token more often
    low_temp_samples = [
        sampler._temperature_sample(logits, temperature=0.1)
        for _ in range(100)
    ]
    low_temp_freq = (torch.cat(low_temp_samples) == 0).float().mean()
    
    # High temp should be more random
    high_temp_samples = [
        sampler._temperature_sample(logits, temperature=2.0)
        for _ in range(100)
    ]
    high_temp_freq = (torch.cat(high_temp_samples) == 0).float().mean()
    
    assert low_temp_freq > high_temp_freq

def test_top_p_dynamic_cutoff():
    sampler = TextSampler(strategy=SamplingStrategy.TOP_P, top_p=0.9)
    
    # Peaked distribution should have small nucleus
    peaked_logits = torch.tensor([[10.0, 1.0, 0.5, 0.1]])
    # Flat distribution should have large nucleus
    flat_logits = torch.tensor([[2.0, 1.9, 1.8, 1.7]])
    
    # Verify nucleus size adapts
    # (would need to modify code to return nucleus size for testing)
```

### 1.6 Token Embeddings & Positional Encoding ⏳

**Status**: ⏳ Pending

**File**: `src/ml/embeddings.py`

**Priority**: MEDIUM-HIGH

**Tasks**:
- [ ] Implement token embeddings
- [ ] Implement sinusoidal positional encoding
- [ ] Implement learnable positional embeddings
- [ ] Add embedding dropout
- [ ] Layer normalization

**Mathematical Foundation**:

**Token Embeddings**: Map discrete tokens to continuous vectors
$$\text{Token Embedding}: \mathbb{N} \rightarrow \mathbb{R}^{d_{model}}$$

**Positional Encoding**: Inject position information (Transformers have no inherent position awareness)

**Sinusoidal Encoding** (Vaswani et al., 2017):
$$
\begin{align}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{align}
$$

Where:
- $pos$: Position in sequence
- $i$: Dimension index
- $d_{model}$: Model dimension

**Properties**:
- Deterministic (no learning required)
- Handles variable sequence lengths
- Relative position: $PE_{pos+k}$ is linear function of $PE_{pos}$

**Implementation**:
```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Token embedding layer
    
    Maps token IDs to dense vectors
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs (batch, seq_len)
        
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        # Scale by sqrt(d_model) as in original paper
        return self.embedding(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017)
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (batch, seq_len, d_model)
        
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (alternative to sinusoidal)
    
    Used in BERT, GPT-2, etc.
    """
    
    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (batch, seq_len, d_model)
        
        Returns:
            x + learned positional embeddings
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        # Get positional embeddings
        pos_embeddings = self.embedding(positions)
        
        # Add to token embeddings
        x = x + pos_embeddings
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer: Token + Positional + Dropout + LayerNorm
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        positional_encoding: str = "sinusoidal",  # or "learned"
        dropout: float = 0.1,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size, d_model, padding_idx
        )
        
        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model, max_len, dropout
            )
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEmbedding(
                max_len, d_model, dropout
            )
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Layer normalization (optional, used in some models)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs (batch, seq_len)
        
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        # Token embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x
```

**Acceptance Criteria**:
- ✅ Token embeddings work
- ✅ Sinusoidal PE produces correct values
- ✅ Learned PE trains properly
- ✅ Can handle variable sequence lengths
- ✅ Dropout applied correctly

---

### 1.7 Configuration & Vector Store ⏳

**Status**: ⏳ Pending (LOWER PRIORITY - After ML core)

**Files**: 
- `src/config/settings.py`
- `src/core/vector_store.py`

(Keeping original sections but deprioritized - these are infrastructure not ML core)

**Features**:
```python
class VectorStore:
    """ChromaDB wrapper for vector storage"""
    
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
    
    def create_collection(self, name: str) -> Collection:
        """Create or get collection"""
        pass
    
    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """Add documents to collection"""
        pass
    
    def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict:
        """Semantic search"""
        pass
    
    def delete_collection(self, name: str):
        """Delete entire collection"""
        pass
```

**Collections**:
- `documents` - PDF, DOCX, TXT files
- `images` - Image analyses
- `audio` - Audio transcriptions
- `web` - Web scraping results

**Acceptance Criteria**:
- ✅ Persistent storage works
- ✅ CRUD operations functional
- ✅ Semantic search returns relevant results
- ✅ Metadata filtering works
- ✅ Multiple collections supported

### 1.7 Utilities Implementation ⏳

**Status**: ⏳ Pending

**Files**: 
- `src/utils/file_handlers.py`
- `src/utils/chunking.py`
- `src/utils/validators.py`

**Tasks**:
- [ ] File type detection
- [ ] File readers (PDF, DOCX, TXT)
- [ ] Intelligent chunking
- [ ] Input validation
- [ ] Error handling utilities

**Key Functions**:

**File Handlers**:
```python
def detect_file_type(file_path: str) -> str:
    """Detect file type from extension/magic bytes"""
    pass

def read_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    pass

def read_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    pass

def read_text(file_path: str) -> str:
    """Read plain text file"""
    pass
```

**Chunking**:
```python
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    strategy: str = "paragraph"
) -> List[str]:
    """Smart text chunking"""
    pass

def semantic_chunking(
    text: str,
    similarity_threshold: float = 0.7
) -> List[str]:
    """Semantic-aware chunking"""
    pass
```

**Validators**:
```python
def validate_file_size(file_size: int, max_size: int):
    """Validate file size"""
    pass

def validate_file_type(file_type: str, allowed_types: List[str]):
    """Validate file type"""
    pass

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for security"""
    pass
```

**Acceptance Criteria**:
- ✅ All file types readable
- ✅ Chunking preserves context
- ✅ Validation prevents issues
- ✅ Error handling robust

---

## PHASE 2: Transformer Architecture & Complete Implementation

**Goal**: Implementarea arhitecturii Transformer complete și a Feed-Forward Networks

**Duration**: 2 săptămâni (10-14 zile)

**Status**: ⏳ Pending

**Dependencies**: Phase 1 complete (Attention, KV Cache, Embeddings)

**Focus**: Build complete Transformer encoder/decoder blocks and assemble full model

### 2.1 Feed-Forward Networks ⏳

**Status**: ⏳ Pending

**File**: `src/ml/transformer.py`

**Priority**: CRITICAL

**Tasks**:
- [ ] Implement position-wise Feed-Forward Network
- [ ] Add GELU/ReLU activation
- [ ] Implement dropout and residual connections
- [ ] Add Layer Normalization
- [ ] Optimize memory usage

**Mathematical Foundation**:

**Feed-Forward Network** (applied position-wise):
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU activation:
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Typical Architecture**:
```
Input (d_model) → Linear (d_ff) → GELU → Linear (d_model) → Output
```

Where $d_{ff} = 4 \times d_{model}$ (expansion factor)

**Example**: GPT-3
- $d_{model} = 12288$
- $d_{ff} = 49152$ (4x expansion)
- Parameters: $2 \times d_{model} \times d_{ff} \approx 1.2B$ per layer!

**Implementation**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Applied independently to each position (token)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            d_model: Model dimension (input/output)
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu")
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        # First linear + activation
        x = self.linear1(x)          # (batch, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear
        x = self.linear2(x)          # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture:
        Input → LayerNorm → MultiHeadAttention → Residual →
        LayerNorm → FeedForward → Residual → Output
    
    Note: Pre-LN (LayerNorm before) vs Post-LN (after) - modern models use Pre-LN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True
    ):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
        self.pre_norm = pre_norm
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            (batch, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-LN: Norm → Sublayer → Residual (modern, more stable)
            
            # Self-attention block
            residual = x
            x = self.norm1(x)
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = residual + self.dropout(attn_output)
            
            # Feed-forward block
            residual = x
            x = self.norm2(x)
            ffn_output = self.ffn(x)
            x = residual + self.dropout(ffn_output)
        else:
            # Post-LN: Sublayer → Residual → Norm (original Transformer)
            
            # Self-attention block
            residual = x
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(residual + self.dropout(attn_output))
            
            # Feed-forward block
            residual = x
            ffn_output = self.ffn(x)
            x = self.norm2(residual + self.dropout(ffn_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer (for autoregressive generation)
    
    Architecture:
        Input → LayerNorm → Masked Self-Attention → Residual →
        LayerNorm → Cross-Attention (optional) → Residual →
        LayerNorm → FeedForward → Residual → Output
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True
    ):
        super().__init__()
        
        # Masked Self-Attention (causal)
        self.self_attn = MultiHeadAttentionWithCache(d_model, n_heads, dropout)
        
        # Cross-Attention (for encoder-decoder models)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x: Decoder input (batch, seq_len, d_model)
            encoder_output: Encoder output for cross-attention (batch, src_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention
            cache: KV cache for generation
            use_cache: Whether to use/return cache
        
        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Updated KV cache
        """
        # Masked self-attention
        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        
        attn_output, new_cache = self.self_attn(
            x, x, x,
            mask=self_attn_mask,
            cache=cache,
            use_cache=use_cache
        )
        x = residual + self.dropout(attn_output)
        
        if not self.pre_norm:
            x = self.norm1(x)
        
        # Cross-attention (if encoder output provided)
        if encoder_output is not None:
            residual = x
            if self.pre_norm:
                x = self.norm2(x)
            
            cross_attn_output, _ = self.cross_attn(
                x, encoder_output, encoder_output,
                mask=cross_attn_mask
            )
            x = residual + self.dropout(cross_attn_output)
            
            if not self.pre_norm:
                x = self.norm2(x)
        
        # Feed-forward
        residual = x
        if self.pre_norm:
            x = self.norm3(x)
        
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        
        if not self.pre_norm:
            x = self.norm3(x)
        
        return x, new_cache


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder layers (e.g., BERT)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len,
            positional_encoding="learned",
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            mask: Padding mask
        
        Returns:
            (batch, seq_len, d_model)
        """
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm
        x = self.norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder layers (e.g., GPT)
    
    Autoregressive language model
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len,
            positional_encoding="learned",
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (share embedding and output projection)
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[Dict[int, KVCache]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[int, KVCache]]]:
        """
        Args:
            input_ids: (batch, seq_len)
            cache: Dict mapping layer index to KV cache
            use_cache: Whether to use/return cache
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_cache: Updated cache dict
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Create causal mask
        mask = create_causal_mask(seq_len, device)
        
        # Initialize cache dict
        if cache is None and use_cache:
            cache = {}
        
        new_cache = {}
        
        # Pass through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = cache.get(layer_idx) if cache else None
            
            x, layer_new_cache = layer(
                x,
                encoder_output=None,
                self_attn_mask=mask,
                cache=layer_cache,
                use_cache=use_cache
            )
            
            if use_cache:
                new_cache[layer_idx] = layer_new_cache
        
        # Final norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        if use_cache:
            return logits, new_cache
        return logits, None
```

**Key Design Choices**:

1. **Pre-LN vs Post-LN**:
   - Pre-LN: Norm before sublayer (GPT-3, modern models) → more stable training
   - Post-LN: Norm after sublayer (original Transformer) → harder to train deep models

2. **Residual Connections**: Critical for gradient flow in deep models

3. **Weight Tying**: Share embedding and output projection weights → reduces parameters, improves quality

4. **GELU vs ReLU**:
   - GELU: Smoother, used in BERT/GPT → better performance
   - ReLU: Faster, simpler

**Acceptance Criteria**:
- ✅ Feed-forward network correct
- ✅ Encoder layer works
- ✅ Decoder layer works with KV cache
- ✅ Full encoder model trains
- ✅ Full decoder model generates text
- ✅ Residual connections work
- ✅ Layer norm applied correctly

### 2.2 Complete Language Model Assembly ⏳

**Status**: ⏳ Pending

**Tasks**:
- [ ] Assemble complete GPT-style model
- [ ] Add language modeling head
- [ ] Implement loss computation
- [ ] Add model initialization
- [ ] Weight tying

(Implementation continues in file...)

---

## PHASE 3: Model Training & Fine-tuning (Optional - can use APIs instead)

**Note**: Phases 3-6 are LOWER PRIORITY. For practical NOVA system, we can use Anthropic/Mistral APIs instead of training from scratch. These phases are kept for completeness and learning purposes.

### 3.1 Basic Agent & RAG (MOVED UP IN PRIORITY)

**Status**: ⏳ Pending

**Goal**: Build simple document agent that uses API-based LLMs + vector store

(Keeping simplified agent implementation - not full specialization)

**Pipeline**:
```
Input: File path → Extract text → Clean → Chunk → Embed → Store → Return metadata
```

**Features**:
```python
class DocumentAgent(BaseAgent):
    """Process and analyze documents (PDF, DOCX, TXT)"""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Full document processing pipeline
        
        Returns:
            {
                'status': 'success',
                'file_name': str,
                'num_chunks': int,
                'total_tokens': int,
                'collection': str,
                'document_ids': List[str]
            }
        """
        pass
    
    def extract_text(self, file_path: str) -> str:
        """Extract text based on file type"""
        pass
    
    def process_and_store(self, text: str, metadata: Dict) -> List[str]:
        """Chunk, embed, and store"""
        pass
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        n_results: int = 5,
        document_filter: Dict = None
    ) -> List[Dict]:
        """Retrieve relevant document chunks"""
        pass
```

**Advanced Features**:
- Hierarchical indexing (document → section → chunk)
- Page number tracking for citations
- Table extraction (future)
- Image extraction from PDFs (future)

**Acceptance Criteria**:
- ✅ Can process PDF files
- ✅ Can process DOCX files
- ✅ Can process TXT files
- ✅ Chunking preserves context
- ✅ Embeddings stored correctly
- ✅ Retrieval returns relevant chunks
- ✅ Metadata includes page numbers

**Testing**:
- Test with small document (< 5 pages)
- Test with large document (> 100 pages)
- Test with various formats
- Verify chunking quality
- Verify retrieval accuracy

### 2.3 Vision Agent 👁️

**Status**: ⏳ Pending

**File**: `src/agents/vision_agent.py`

**Priority**: High

**Tasks**:
- [ ] Implement image loading
- [ ] Claude Vision integration
- [ ] OCR text extraction
- [ ] Detailed visual analysis
- [ ] Embedding of analysis text
- [ ] Storage in vector DB

**Features**:
```python
class VisionAgent(BaseAgent):
    """Analyze images and extract information"""
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Full image analysis pipeline
        
        Returns:
            {
                'status': 'success',
                'image_name': str,
                'description': str,
                'detected_text': str,
                'objects': List[str],
                'analysis': str,
                'embedding_id': str
            }
        """
        pass
    
    def analyze_image(
        self,
        image_path: str,
        custom_prompt: str = None
    ) -> str:
        """Analyze image with Claude Vision"""
        pass
    
    def extract_text_ocr(self, image_path: str) -> str:
        """Extract text from image (OCR)"""
        pass
    
    def store_analysis(
        self,
        analysis: str,
        metadata: Dict
    ) -> str:
        """Store analysis in vector DB"""
        pass
```

**Analysis Prompt Template**:
```python
VISION_ANALYSIS_PROMPT = """
Analyze this image in comprehensive detail:

1. **Visual Description**: Describe what you see (objects, people, scenery, layout)
2. **Text Content (OCR)**: Extract ALL visible text exactly as it appears
3. **Context & Meaning**: What is the purpose/message of this image?
4. **Technical Details**: Colors, composition, quality, style
5. **Relevant Information**: Any other notable details

Be thorough and specific in your analysis.
"""
```

**Acceptance Criteria**:
- ✅ Can analyze JPG, PNG, WEBP images
- ✅ OCR extracts text accurately
- ✅ Visual description is detailed
- ✅ Analysis stored in vector DB
- ✅ Can retrieve similar images

**Testing**:
- Test with photos
- Test with screenshots
- Test with diagrams/charts
- Test OCR accuracy
- Verify analysis quality

### 2.4 Audio Agent 🎵

**Status**: ⏳ Pending

**File**: `src/agents/audio_agent.py`

**Priority**: Medium

**Tasks**:
- [ ] Implement audio loading
- [ ] Faster-Whisper integration
- [ ] Transcription with timestamps
- [ ] Speaker diarization (if possible)
- [ ] Text processing and chunking
- [ ] Storage in vector DB

**Features**:
```python
class AudioAgent(BaseAgent):
    """Transcribe and analyze audio files"""
    
    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Full audio processing pipeline
        
        Returns:
            {
                'status': 'success',
                'audio_name': str,
                'duration': float,
                'language': str,
                'transcription': str,
                'segments': List[Dict],
                'embedding_ids': List[str]
            }
        """
        pass
    
    def transcribe(
        self,
        audio_path: str,
        language: str = None
    ) -> Dict:
        """Transcribe audio with Faster-Whisper"""
        pass
    
    def process_transcription(
        self,
        transcription: Dict,
        metadata: Dict
    ) -> List[str]:
        """Process and store transcription"""
        pass
```

**Faster-Whisper Configuration**:
```python
model = WhisperModel(
    "large-v3",  # Best quality
    device="cpu",  # or "cuda" if available
    compute_type="int8"  # Optimized for CPU
)
```

**Acceptance Criteria**:
- ✅ Can transcribe MP3, WAV, M4A
- ✅ Transcription is accurate
- ✅ Timestamps preserved
- ✅ Language auto-detection works
- ✅ Stored in vector DB

**Testing**:
- Test with English audio
- Test with Romanian audio
- Test with background noise
- Verify accuracy
- Check performance

### 2.5 Web Agent 🌐

**Status**: ⏳ Pending

**File**: `src/agents/web_agent.py`

**Priority**: Medium-Low

**Tasks**:
- [ ] Implement URL fetching
- [ ] HTML parsing (BeautifulSoup)
- [ ] Content extraction
- [ ] Clean and format text
- [ ] Chunking and embedding
- [ ] Storage with source tracking

**Features**:
```python
class WebAgent(BaseAgent):
    """Scrape and analyze web content"""
    
    def process(self, url: str) -> Dict[str, Any]:
        """
        Full web scraping pipeline
        
        Returns:
            {
                'status': 'success',
                'url': str,
                'title': str,
                'content': str,
                'num_chunks': int,
                'embedding_ids': List[str]
            }
        """
        pass
    
    def fetch_url(self, url: str) -> str:
        """Fetch HTML content"""
        pass
    
    def extract_content(self, html: str) -> Dict:
        """Extract title, text, links"""
        pass
    
    def clean_text(self, text: str) -> str:
        """Remove JS, CSS, extra whitespace"""
        pass
```

**Content Extraction Strategy**:
- Extract `<title>`
- Extract main content (heuristic: longest text block)
- Remove navigation, footer, sidebar
- Preserve structure (headers, paragraphs)

**Acceptance Criteria**:
- ✅ Can fetch and parse HTML
- ✅ Content extraction is clean
- ✅ Stores URL source
- ✅ Handles errors gracefully

**Testing**:
- Test with news articles
- Test with documentation pages
- Test with various websites
- Verify content quality

---

## PHASE 3: Orchestrator & Integration

**Goal**: Coordonarea agenților și integrarea sistemului complet

**Duration**: 1 săptămână (5-7 zile)

**Status**: ⏳ Pending

**Dependencies**: Phase 2 complete

### 3.1 Orchestrator Implementation 🎯

**Status**: ⏳ Pending

**File**: `src/agents/orchestrator.py`

**Priority**: Critical

**Tasks**:
- [ ] Implement routing logic
- [ ] Multi-agent coordination
- [ ] Context management
- [ ] Result synthesis
- [ ] Error handling

**Features**:
```python
class Orchestrator:
    """Coordinate multiple agents and synthesize results"""
    
    def __init__(
        self,
        document_agent: DocumentAgent,
        vision_agent: VisionAgent,
        audio_agent: AudioAgent,
        web_agent: WebAgent,
        llm_interface: LLMInterface
    ):
        self.agents = {
            'document': document_agent,
            'vision': vision_agent,
            'audio': audio_agent,
            'web': web_agent
        }
        self.llm = llm_interface
        self.conversation_history = []
    
    def process_query(
        self,
        query: str,
        files: List[str] = None,
        urls: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        
        Pipeline:
        1. Analyze query intent
        2. Route to appropriate agents
        3. Collect results
        4. Synthesize final response
        """
        pass
    
    def route_to_agents(
        self,
        query: str,
        files: List[str],
        urls: List[str]
    ) -> Dict[str, Any]:
        """Determine which agents to call"""
        pass
    
    def synthesize_response(
        self,
        query: str,
        agent_results: Dict[str, Any],
        retrieved_context: List[Dict]
    ) -> str:
        """Synthesize final response from multiple sources"""
        pass
```

**Routing Logic**:
```python
def route_to_agents(self, query, files, urls):
    agents_to_call = []
    
    # File-based routing
    for file in files:
        file_type = detect_file_type(file)
        if file_type in ['pdf', 'docx', 'txt']:
            agents_to_call.append(('document', file))
        elif file_type in ['jpg', 'png', 'webp']:
            agents_to_call.append(('vision', file))
        elif file_type in ['mp3', 'wav', 'm4a']:
            agents_to_call.append(('audio', file))
    
    # URL-based routing
    if urls:
        for url in urls:
            agents_to_call.append(('web', url))
    
    # Query-based routing (keywords)
    if 'search' in query.lower() or 'web' in query.lower():
        # Consider web search
        pass
    
    return agents_to_call
```

**Synthesis Strategy**:
```python
SYNTHESIS_PROMPT = """
You are a helpful AI assistant. Based on the following information sources, provide a comprehensive answer to the user's question.

Sources:
{sources}

User Question: {query}

Instructions:
1. Synthesize information from all sources
2. Cite sources when making claims [Source: filename.pdf]
3. If sources conflict, mention both perspectives
4. Be clear and concise
5. If sources don't fully answer the question, say so

Answer:
"""
```

**Acceptance Criteria**:
- ✅ Routes to correct agents
- ✅ Handles multiple file types
- ✅ Synthesizes coherent responses
- ✅ Cites sources properly
- ✅ Handles errors gracefully

### 3.2 Context Management 💾

**Status**: ⏳ Pending

**Tasks**:
- [ ] Implement conversation history
- [ ] Context window management
- [ ] Relevant context retrieval
- [ ] Session management

**Features**:
```python
class ContextManager:
    """Manage conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.max_history = max_history
    
    def add_interaction(
        self,
        query: str,
        response: str,
        sources: List[str]
    ):
        """Add interaction to history"""
        pass
    
    def get_relevant_context(
        self,
        current_query: str
    ) -> str:
        """Get relevant past context for current query"""
        pass
    
    def build_context_window(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        conversation_context: str
    ) -> str:
        """Build complete context for LLM"""
        pass
```

**Acceptance Criteria**:
- ✅ History maintained correctly
- ✅ Context window stays within limits
- ✅ Relevant context retrieved
- ✅ Old context pruned appropriately

### 3.3 RAG Pipeline Integration 🔗

**Status**: ⏳ Pending

**Tasks**:
- [ ] Implement complete RAG flow
- [ ] Query expansion
- [ ] Hybrid search (optional)
- [ ] Re-ranking (optional)
- [ ] Result fusion

**RAG Pipeline**:
```
Query → Embed → Vector Search → Re-rank → Context Construction → LLM Generation → Response
```

**Advanced Features** (optional):
- **Query Expansion**: Generate multiple query variations
- **Hybrid Search**: Combine semantic + keyword search
- **Re-ranking**: Use cross-encoder for better relevance
- **MMR**: Maximum Marginal Relevance for diversity

**Acceptance Criteria**:
- ✅ Complete RAG pipeline works
- ✅ Retrieval is relevant
- ✅ Responses cite sources
- ✅ Quality is high

---

## PHASE 4: User Interface & Experience

**Goal**: Crearea interfeței utilizator cu Streamlit

**Duration**: 1 săptămână (5-7 zile)

**Status**: ⏳ Pending

**Dependencies**: Phase 3 complete

### 4.1 Basic Streamlit Interface 📱

**Status**: ⏳ Pending

**File**: `src/ui/streamlit_app.py`

**Tasks**:
- [ ] Create main app structure
- [ ] Implement chat interface
- [ ] File upload functionality
- [ ] URL input
- [ ] Session state management

**Features**:
```python
import streamlit as st

def main():
    st.set_page_config(
        page_title="NOVA AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        # API key inputs (if needed)
        # File uploads
        # Settings
    
    # Main chat interface
    st.title("🤖 NOVA AI Assistant")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Process query
        # Display response
        pass
```

**UI Components**:
1. **Chat Interface**:
   - Message history display
   - User input box
   - Streaming responses

2. **File Upload Panel**:
   - Multi-file upload
   - File type validation
   - Upload progress
   - File preview

3. **Configuration Sidebar**:
   - Model selection
   - Temperature control
   - Max tokens
   - Clear history button

4. **Status Indicators**:
   - Processing indicators
   - Token usage
   - Cost estimation
   - Error messages

**Acceptance Criteria**:
- ✅ Chat interface functional
- ✅ File upload works
- ✅ Responses display correctly
- ✅ Session state persists
- ✅ UI is responsive

### 4.2 Advanced UI Features ✨

**Status**: ⏳ Pending

**Tasks**:
- [ ] Streaming responses
- [ ] Source citations display
- [ ] File preview
- [ ] Export conversation
- [ ] Dark/light theme

**Advanced Features**:
```python
# Streaming response
def stream_response(response_text):
    for chunk in response_text.split():
        yield chunk + " "
        time.sleep(0.05)

# Source citations
def display_sources(sources):
    with st.expander("📚 Sources"):
        for source in sources:
            st.markdown(f"- **{source['name']}** (page {source['page']})")

# File preview
def preview_file(file):
    if file.type == "application/pdf":
        # Show first page or thumbnail
        pass
    elif file.type.startswith("image"):
        st.image(file)
```

**Acceptance Criteria**:
- ✅ Streaming works smoothly
- ✅ Sources displayed clearly
- ✅ File preview functional
- ✅ Export works
- ✅ Theme switching works

### 4.3 Error Handling & UX Polish 🎨

**Status**: ⏳ Pending

**Tasks**:
- [ ] User-friendly error messages
- [ ] Loading states
- [ ] Input validation feedback
- [ ] Help tooltips
- [ ] Keyboard shortcuts

**Error Handling**:
```python
try:
    response = orchestrator.process_query(query, files)
except APIError as e:
    st.error(f"API Error: {str(e)}")
except ValidationError as e:
    st.warning(f"Invalid input: {str(e)}")
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    logger.exception("Unexpected error in UI")
```

**Loading States**:
```python
with st.spinner("Processing your request..."):
    response = process_query(query)

# Or more detailed:
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("Analyzing files...")
progress_bar.progress(25)

status_text.text("Searching knowledge base...")
progress_bar.progress(50)

status_text.text("Generating response...")
progress_bar.progress(75)

progress_bar.progress(100)
status_text.text("Done!")
```

**Acceptance Criteria**:
- ✅ Errors displayed clearly
- ✅ Loading states informative
- ✅ Validation provides feedback
- ✅ Tooltips helpful
- ✅ UX feels polished

---

## PHASE 5: Testing & Optimization

**Goal**: Testare extensivă și optimizări de performanță

**Duration**: 1 săptămână (5-7 zile)

**Status**: ⏳ Pending

**Dependencies**: Phase 4 complete

### 5.1 Unit Testing 🧪

**Status**: ⏳ Pending

**Tasks**:
- [ ] Test core utilities
- [ ] Test embeddings generation
- [ ] Test vector store operations
- [ ] Test each agent independently
- [ ] Test LLM interface

**Testing Framework**: pytest

**Test Structure**:
```
tests/
├── test_core/
│   ├── test_llm_interface.py
│   ├── test_embeddings.py
│   └── test_vector_store.py
├── test_agents/
│   ├── test_document_agent.py
│   ├── test_vision_agent.py
│   ├── test_audio_agent.py
│   └── test_web_agent.py
├── test_utils/
│   ├── test_chunking.py
│   ├── test_file_handlers.py
│   └── test_validators.py
└── conftest.py  # Shared fixtures
```

**Example Tests**:
```python
# test_embeddings.py
def test_embedding_generation():
    embedder = EmbeddingGenerator()
    text = "This is a test sentence."
    embedding = embedder.generate_embedding(text)
    
    assert embedding.shape == (1024,)
    assert np.linalg.norm(embedding) - 1.0 < 1e-5  # L2 normalized

def test_embedding_similarity():
    embedder = EmbeddingGenerator()
    emb1 = embedder.generate_embedding("cat")
    emb2 = embedder.generate_embedding("feline")
    emb3 = embedder.generate_embedding("car")
    
    sim_12 = embedder.similarity(emb1, emb2)
    sim_13 = embedder.similarity(emb1, emb3)
    
    assert sim_12 > sim_13  # cat-feline more similar than cat-car
```

**Coverage Target**: > 80%

**Acceptance Criteria**:
- ✅ All critical functions tested
- ✅ Tests pass consistently
- ✅ Coverage > 80%
- ✅ Edge cases covered

### 5.2 Integration Testing 🔗

**Status**: ⏳ Pending

**Tasks**:
- [ ] Test end-to-end workflows
- [ ] Test multi-agent coordination
- [ ] Test RAG pipeline
- [ ] Test UI interactions

**Integration Test Scenarios**:

1. **Document Processing**:
   - Upload PDF → Process → Query → Verify response

2. **Multi-Modal**:
   - Upload PDF + Image → Query → Verify synthesis

3. **Conversation Flow**:
   - Multiple queries → Verify context maintained

4. **Error Recovery**:
   - Invalid files → Verify graceful handling

**Acceptance Criteria**:
- ✅ All workflows work end-to-end
- ✅ Multi-agent coordination smooth
- ✅ Context maintained across queries
- ✅ Errors handled gracefully

### 5.3 Performance Optimization ⚡

**Status**: ⏳ Pending

**Tasks**:
- [ ] Profile slow operations
- [ ] Optimize embedding generation
- [ ] Implement caching
- [ ] Batch API calls
- [ ] Optimize vector search

**Optimization Targets**:

| Operation | Current | Target | Strategy |
|-----------|---------|--------|----------|
| Document Processing | ? | < 30s | Batch embeddings, parallel chunking |
| Query Response | ? | < 5s | Caching, optimized retrieval |
| Embedding Generation | ? | < 2s | Batch API calls, caching |
| Vector Search | ? | < 1s | HNSW optimization |

**Caching Strategy**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_embedding_cached(text: str) -> np.ndarray:
    """Cached embedding generation"""
    return embedding_generator.generate_embedding(text)
```

**Batch Processing**:
```python
# Instead of:
for chunk in chunks:
    embedding = get_embedding(chunk)  # N API calls

# Do:
embeddings = get_embeddings_batch(chunks)  # 1 API call
```

**Acceptance Criteria**:
- ✅ Response time < 5s for queries
- ✅ Document processing < 30s
- ✅ Caching reduces API calls by 50%+
- ✅ Memory usage reasonable

### 5.4 Quality Assurance 📊

**Status**: ⏳ Pending

**Tasks**:
- [ ] Evaluate retrieval quality
- [ ] Evaluate response quality
- [ ] Test with real use cases
- [ ] Collect metrics

**Evaluation Metrics**:

1. **Retrieval Metrics**:
   - Precision@5: Relevant chunks in top 5
   - Recall@5: Coverage of relevant info
   - MRR: Mean Reciprocal Rank

2. **Response Quality**:
   - Faithfulness: Response matches sources
   - Relevance: Response answers query
   - Coherence: Response is well-structured
   - Citation Accuracy: Sources cited correctly

**Evaluation Framework**:
```python
def evaluate_rag_system(test_cases: List[TestCase]):
    """
    test_case = {
        'query': str,
        'ground_truth_answer': str,
        'relevant_documents': List[str]
    }
    """
    results = {
        'precision': [],
        'recall': [],
        'faithfulness': [],
        'relevance': []
    }
    
    for case in test_cases:
        # Run query
        response = orchestrator.process_query(case['query'])
        
        # Evaluate retrieval
        retrieved_docs = response['sources']
        precision = calculate_precision(
            retrieved_docs,
            case['relevant_documents']
        )
        
        # Evaluate response (using LLM as judge)
        faithfulness = llm_judge_faithfulness(
            response['answer'],
            retrieved_docs
        )
        
        results['precision'].append(precision)
        results['faithfulness'].append(faithfulness)
    
    return {k: np.mean(v) for k, v in results.items()}
```

**Quality Targets**:
- Precision@5 > 0.8
- Faithfulness > 0.9
- Relevance > 0.85

**Acceptance Criteria**:
- ✅ Metrics collected systematically
- ✅ Quality meets targets
- ✅ Issues documented
- ✅ Improvement plan created

---

## PHASE 6: Advanced Features

**Goal**: Implementarea feature-urilor avansate și îmbunătățiri

**Duration**: 2 săptămâni (10-14 zile)

**Status**: ⏳ Pending

**Dependencies**: Phase 5 complete

### 6.1 Advanced RAG Features 🚀

**Status**: ⏳ Pending

**Priority**: Medium

**Tasks**:
- [ ] Implement query expansion
- [ ] Add hybrid search (semantic + BM25)
- [ ] Implement re-ranking with cross-encoder
- [ ] Add MMR for diversity
- [ ] Implement RAG fusion

**Features**:

**1. Query Expansion**:
```python
def expand_query(query: str) -> List[str]:
    """Generate multiple query variations"""
    expansion_prompt = f"""
    Generate 3 alternative phrasings of this query:
    "{query}"
    
    Variations should:
    - Use different words but same meaning
    - Cover different aspects of the question
    - Be concise
    """
    
    variations = llm.generate(expansion_prompt)
    return [query] + parse_variations(variations)
```

**2. Hybrid Search**:
```python
def hybrid_search(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Document]:
    """Combine semantic and keyword search"""
    
    # Semantic search
    semantic_results = vector_store.search(query, n_results=20)
    
    # BM25 keyword search
    keyword_results = bm25_index.search(query, top_k=20)
    
    # Combine scores
    combined = fuse_results(
        semantic_results,
        keyword_results,
        semantic_weight,
        keyword_weight
    )
    
    return combined[:5]
```

**3. Re-ranking**:
```python
def rerank_results(
    query: str,
    candidates: List[Document],
    top_k: int = 5
) -> List[Document]:
    """Re-rank using cross-encoder"""
    
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    pairs = [(query, doc.text) for doc in candidates]
    scores = model.predict(pairs)
    
    # Sort by score
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [doc for doc, score in ranked[:top_k]]
```

**4. MMR (Maximum Marginal Relevance)**:
```python
def mmr_rerank(
    query_embedding: np.ndarray,
    candidates: List[Document],
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[Document]:
    """Select diverse relevant documents"""
    
    selected = []
    remaining = candidates.copy()
    
    # Select first (most relevant)
    first = max(remaining, key=lambda d: cosine_sim(query_embedding, d.embedding))
    selected.append(first)
    remaining.remove(first)
    
    # Iteratively select diverse documents
    while len(selected) < top_k and remaining:
        mmr_scores = []
        for doc in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, doc.embedding)
            
            # Max similarity to already selected
            max_sim = max([
                cosine_sim(doc.embedding, s.embedding)
                for s in selected
            ])
            
            # MMR score: balance relevance and diversity
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((doc, mmr))
        
        # Select highest MMR
        next_doc = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(next_doc)
        remaining.remove(next_doc)
    
    return selected
```

**Acceptance Criteria**:
- ✅ Query expansion improves recall
- ✅ Hybrid search outperforms pure semantic
- ✅ Re-ranking improves precision
- ✅ MMR provides diversity

### 6.2 Memory & Personalization 🧠

**Status**: ⏳ Pending

**Priority**: Low-Medium

**Tasks**:
- [ ] Implement user profiles
- [ ] Long-term memory storage
- [ ] Preference learning
- [ ] Personalized responses

**Features**:

**User Profile**:
```python
class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences = {}
        self.interaction_history = []
        self.frequently_queried_topics = []
    
    def update_preferences(self, interaction: Dict):
        """Learn from user interactions"""
        pass
    
    def get_personalized_context(self, query: str) -> str:
        """Get context based on user history"""
        pass
```

**Long-term Memory**:
```python
class MemoryManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.collection_name = "user_memory"
    
    def store_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        timestamp: datetime
    ):
        """Store interaction in long-term memory"""
        pass
    
    def retrieve_relevant_memories(
        self,
        user_id: str,
        current_query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """Retrieve relevant past interactions"""
        pass
```

**Acceptance Criteria**:
- ✅ User profiles persist
- ✅ Memory retrieval works
- ✅ Personalization improves UX

### 6.3 Multi-User Support 👥

**Status**: ⏳ Pending

**Priority**: Low

**Tasks**:
- [ ] Add authentication
- [ ] User session management
- [ ] Data isolation per user
- [ ] User management UI

**Features**:
- Simple authentication (username/password)
- Session management with Streamlit
- Separate vector DB collections per user
- Admin panel for user management

**Acceptance Criteria**:
- ✅ Multiple users can use system
- ✅ Data is isolated
- ✅ Sessions managed correctly

### 6.4 Advanced Analytics 📈

**Status**: ⏳ Pending

**Priority**: Low

**Tasks**:
- [ ] Usage tracking
- [ ] Cost monitoring
- [ ] Performance dashboards
- [ ] Quality metrics visualization

**Features**:
```python
class AnalyticsTracker:
    def track_query(
        self,
        query: str,
        response_time: float,
        tokens_used: int,
        cost: float,
        user_id: str
    ):
        """Track query metrics"""
        pass
    
    def get_usage_stats(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Get usage statistics"""
        return {
            'total_queries': int,
            'total_tokens': int,
            'total_cost': float,
            'avg_response_time': float,
            'users_active': int
        }
```

**Dashboard Components**:
- Queries over time
- Cost breakdown by model
- Response time distribution
- Most queried topics
- User activity

**Acceptance Criteria**:
- ✅ All metrics tracked
- ✅ Dashboard visualizes data
- ✅ Insights actionable

---

## TECHNICAL STACK SUMMARY

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Main development language |
| **LLM Provider** | Anthropic | Latest | Claude 3.5 Sonnet for text & vision |
| **LLM Provider** | Mistral | Latest | Alternative LLM + embeddings |
| **Embeddings** | Mistral Embed | Latest | 1024D semantic embeddings |
| **Vector DB** | ChromaDB | 0.5.0+ | Local vector storage |
| **UI Framework** | Streamlit | 1.39.0+ | Web interface |
| **Audio** | Faster-Whisper | 1.0.0+ | Local audio transcription |
| **PDF** | PyMuPDF | 1.24.0+ | PDF text extraction |
| **Document** | python-docx | 1.1.0+ | DOCX processing |
| **Web** | BeautifulSoup4 | 4.12.0+ | HTML parsing |
| **Image** | Pillow | 10.0.0+ | Image processing |
| **Testing** | pytest | Latest | Unit & integration tests |

### API Keys Required

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
```

### System Requirements

**Minimum**:
- Python 3.11+
- 8GB RAM
- 10GB free disk space
- Internet connection (for APIs)

**Recommended**:
- Python 3.11+
- 16GB RAM
- 20GB free disk space
- Good internet connection
- (Optional) GPU for faster Whisper transcription

---

## SUCCESS METRICS

### Phase Completion Metrics

| Phase | Success Criteria | Status |
|-------|-----------------|--------|
| **Phase 1** | - All core components implemented<br>- Configuration working<br>- LLM interface functional<br>- Vector DB operational | 🔄 25% |
| **Phase 2** | - All 4 agents functional<br>- Can process all file types<br>- Storage in vector DB works | ⏳ 0% |
| **Phase 3** | - Orchestrator routes correctly<br>- Multi-agent coordination works<br>- RAG pipeline complete | ⏳ 0% |
| **Phase 4** | - UI functional and responsive<br>- File upload works<br>- Chat interface smooth | ⏳ 0% |
| **Phase 5** | - Test coverage > 80%<br>- Response time < 5s<br>- Quality metrics meet targets | ⏳ 0% |
| **Phase 6** | - Advanced features implemented<br>- System production-ready | ⏳ 0% |

### Overall Project Success Metrics

**Functional Metrics**:
- ✅ Can process PDF, DOCX, TXT documents
- ✅ Can analyze images with OCR
- ✅ Can transcribe audio
- ✅ Can scrape web content
- ✅ Provides coherent synthesized responses
- ✅ Cites sources accurately

**Performance Metrics**:
- Response time < 5 seconds for queries
- Document processing < 30 seconds
- Retrieval precision > 80%
- Response faithfulness > 90%

**Quality Metrics**:
- User satisfaction > 4/5
- Accuracy of information > 85%
- Source citation accuracy > 95%

**Technical Metrics**:
- Test coverage > 80%
- Zero critical bugs
- Code maintainability score > B
- Documentation complete

---

## RISK MANAGEMENT

### Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits | Medium | High | Implement caching, batch requests |
| Poor retrieval quality | Medium | High | Use hybrid search, re-ranking |
| Slow response times | Medium | Medium | Optimize queries, add caching |
| File processing errors | Low | Medium | Robust error handling, validation |
| Cost overruns | Medium | Medium | Monitor usage, set limits |
| ChromaDB scalability | Low | Medium | Plan migration path to Pinecone if needed |

### Contingency Plans

1. **API Issues**: Implement retries with exponential backoff, have fallback models
2. **Performance Issues**: Profile and optimize, consider async processing
3. **Quality Issues**: Implement evaluation framework early, iterate on prompts
4. **Cost Issues**: Set budget alerts, optimize API usage

---

## NEXT IMMEDIATE STEPS

### Week 1 (Current)

**Day 1-2**: ✅ Project setup
- [x] Create directory structure
- [x] Setup virtual environment
- [x] Install dependencies
- [x] Create documentation

**Day 3-4**: 🔄 Core infrastructure
- [ ] Implement configuration management
- [ ] Create LLM interface
- [ ] Setup embeddings
- [ ] Initialize vector store

**Day 5-7**: Core utilities
- [ ] File handlers
- [ ] Chunking strategies
- [ ] Validators
- [ ] Logging setup

### Week 2: Document Agent + Testing

### Week 3-4: Remaining Agents

### Week 5: Integration

### Week 6: UI

### Week 7: Testing & QA

### Week 8: Polish & Advanced Features

---

## CONCLUSION

Acest roadmap oferă o structură clară și sistemică pentru dezvoltarea proiectului NOVA. Fiecare fază construiește pe baza celei anterioare, cu checkpoints clare și criterii de acceptare.

**Key Success Factors**:
1. ✅ Modularitate - fiecare componentă independentă
2. ✅ Testing early - testăm pe măsură ce construim
3. ✅ Iterative development - MVP rapid, apoi îmbunătățiri
4. ✅ Documentation - documentăm continuu
5. ✅ User feedback - testăm cu utilizatori reali

**Current Status**: Phase 1, 25% complete
**Next Milestone**: Complete core infrastructure (Week 1)
**Target MVP Date**: End of Week 5
**Target v1.0 Date**: End of Week 8

---

**Last Updated**: 28 Noiembrie 2025
**Document Owner**: NOVA Development Team
**Status**: 🔄 Active Development
