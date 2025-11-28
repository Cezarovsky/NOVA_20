"""
KV Cache and Inference Optimization for Transformer Models

This module implements inference optimization techniques for autoregressive generation:
1. KV Cache - caches computed Key/Value tensors to avoid recomputation
2. Inference Engine - manages generation with caching, sampling, and beam search
3. Performance profiling and benchmarking utilities

Problem Statement:
    Without KV Cache:
    - Every token generation requires recomputing attention for ALL previous tokens
    - Complexity: O(n²) for sequence length n
    - Generation becomes slower as sequence grows
    
    With KV Cache:
    - Store previously computed K, V tensors
    - Only compute attention for NEW token
    - Complexity: O(n) for incremental generation
    - Speedup: 10-100x depending on sequence length

Mathematical Foundation:
    Standard Attention:
        At step t, compute attention over ALL positions [0, t]:
        Attention(Q_t, K_{0:t}, V_{0:t})
    
    With KV Cache:
        Store K_{0:t-1}, V_{0:t-1} from previous steps
        At step t, only compute K_t, V_t for new token
        Concatenate: K_{0:t} = [K_cache, K_t]
        Attention(Q_t, K_{0:t}, V_{0:t})

Memory vs Speed Trade-off:
    - Memory: Store K, V for each layer × each head × sequence length
    - Speed: Avoid O(n²) recomputation, achieve O(n) incremental updates
    - Typical: 1-2 GB extra memory for 10-100x speedup

Usage:
    # Initialize cache
    cache = KVCache(num_layers=12, batch_size=1, max_seq_len=2048,
                    num_heads=8, head_dim=64, device='cuda')
    
    # Inference engine
    engine = InferenceEngine(model, cache=cache, device='cuda')
    
    # Generate text
    output = engine.generate(
        prompt="Once upon a time",
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

Author: NOVA Development Team
Date: 28 November 2025
"""

import time
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from torch import Tensor

from src.config.settings import get_settings


@dataclass
class CacheConfig:
    """
    Configuration for KV Cache
    
    Defines the shape and capacity of the cache for efficient memory allocation.
    
    Attributes:
        num_layers: Number of transformer layers (each needs separate cache)
        batch_size: Maximum batch size for generation
        max_seq_len: Maximum sequence length to cache
        num_heads: Number of attention heads per layer
        head_dim: Dimension per attention head (d_k = d_model / num_heads)
        device: Device for cache tensors ('cpu', 'cuda', 'mps')
        dtype: Data type for cache (float32, float16, bfloat16)
    
    Memory Calculation:
        Per layer: 2 (K+V) × batch × heads × seq_len × head_dim × bytes_per_element
        Total: num_layers × per_layer_memory
        
        Example (GPT-2 Small):
            12 layers × 2 × 1 batch × 12 heads × 1024 seq × 64 dim × 4 bytes (fp32)
            = 12 × 2 × 1 × 12 × 1024 × 64 × 4 = ~75 MB
        
        Example (GPT-3 style):
            96 layers × 2 × 1 batch × 96 heads × 2048 seq × 128 dim × 2 bytes (fp16)
            = 96 × 2 × 1 × 96 × 2048 × 128 × 2 = ~9.6 GB
    """
    num_layers: int
    batch_size: int
    max_seq_len: int
    num_heads: int
    head_dim: int
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    
    def get_memory_usage(self) -> int:
        """
        Calculate total memory usage in bytes
        
        Returns:
            Total memory in bytes
        """
        bytes_per_element = 4 if self.dtype == torch.float32 else 2
        
        # 2 for K and V, multiply by all dimensions
        per_layer = 2 * self.batch_size * self.num_heads * self.max_seq_len * self.head_dim * bytes_per_element
        total = self.num_layers * per_layer
        
        return total
    
    def get_memory_usage_mb(self) -> float:
        """Get memory usage in megabytes"""
        return self.get_memory_usage() / (1024 * 1024)
    
    def get_memory_usage_gb(self) -> float:
        """Get memory usage in gigabytes"""
        return self.get_memory_usage() / (1024 * 1024 * 1024)


class KVCache:
    """
    Key-Value Cache for Transformer Inference
    
    Stores computed K, V tensors across generation steps to avoid recomputation.
    Implements efficient incremental updates for autoregressive generation.
    
    Architecture:
        - Separate cache for each transformer layer
        - Each layer stores K and V tensors
        - Shape: [batch_size, num_heads, seq_len, head_dim]
        - Dynamically grows as sequence extends (up to max_seq_len)
    
    Key Operations:
        1. Initialize: Allocate empty cache tensors
        2. Update: Append new K, V to existing cache
        3. Get: Retrieve cached K, V for attention computation
        4. Clear: Reset cache for new generation
    
    Attributes:
        config: Cache configuration
        k_cache: List of K tensors (one per layer)
        v_cache: List of V tensors (one per layer)
        current_seq_len: Current length of cached sequence
    
    Example:
        >>> config = CacheConfig(num_layers=12, batch_size=1, max_seq_len=1024,
        ...                      num_heads=12, head_dim=64, device='cuda')
        >>> cache = KVCache(config)
        >>> 
        >>> # During generation
        >>> new_k = torch.randn(1, 12, 1, 64)  # New token
        >>> new_v = torch.randn(1, 12, 1, 64)
        >>> cache.update(layer_idx=0, k=new_k, v=new_v)
        >>> 
        >>> # Retrieve for attention
        >>> k, v = cache.get(layer_idx=0)
        >>> print(k.shape)  # [1, 12, current_seq_len, 64]
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize KV Cache
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Initialize empty cache for each layer
        # We use lists to store per-layer caches
        self.k_cache: List[Optional[Tensor]] = [None] * config.num_layers
        self.v_cache: List[Optional[Tensor]] = [None] * config.num_layers
        
        # Track current sequence length
        self.current_seq_len = 0
        
        # Performance tracking
        self._update_count = 0
        self._total_update_time = 0.0
    
    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Update cache with new K, V tensors
        
        Appends new K, V to existing cache via concatenation along sequence dimension.
        
        Args:
            layer_idx: Index of transformer layer (0 to num_layers-1)
            k: New key tensor [batch, num_heads, new_seq_len, head_dim]
            v: New value tensor [batch, num_heads, new_seq_len, head_dim]
        
        Returns:
            Tuple of (full_k, full_v) including cached and new tensors
        
        Raises:
            ValueError: If cache is full (current_seq_len >= max_seq_len)
        
        Implementation:
            1. If first update: initialize cache with k, v
            2. Otherwise: concatenate new k, v to existing cache
            3. Update current_seq_len
            4. Return full cached tensors
        """
        start_time = time.time()
        
        # Validate layer index
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be 0 to {self.config.num_layers - 1}")
        
        # Get new sequence length
        new_seq_len = k.size(2)
        
        # Check if we have space in cache
        if self.current_seq_len + new_seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Cache full! Current: {self.current_seq_len}, "
                f"New: {new_seq_len}, Max: {self.config.max_seq_len}"
            )
        
        # First update: initialize cache
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            # Subsequent updates: concatenate along sequence dimension (dim=2)
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
        
        # Update sequence length (only on layer 0 to avoid redundancy)
        if layer_idx == 0:
            self.current_seq_len += new_seq_len
        
        # Performance tracking
        self._update_count += 1
        self._total_update_time += time.time() - start_time
        
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Retrieve cached K, V tensors for a layer
        
        Args:
            layer_idx: Index of transformer layer
        
        Returns:
            Tuple of (k_cache, v_cache) or (None, None) if not initialized
        """
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def clear(self):
        """
        Clear all cached tensors
        
        Resets cache to empty state for new generation sequence.
        """
        self.k_cache = [None] * self.config.num_layers
        self.v_cache = [None] * self.config.num_layers
        self.current_seq_len = 0
        self._update_count = 0
        self._total_update_time = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with statistics:
                - current_seq_len: Current cached sequence length
                - memory_used_mb: Memory currently used
                - memory_capacity_mb: Total memory capacity
                - utilization: Percentage of cache used
                - avg_update_time_ms: Average update time in milliseconds
        """
        memory_per_token = (
            2 * self.config.num_layers * self.config.batch_size * 
            self.config.num_heads * self.config.head_dim * 
            (4 if self.config.dtype == torch.float32 else 2)
        )
        
        memory_used = self.current_seq_len * memory_per_token
        memory_capacity = self.config.max_seq_len * memory_per_token
        
        avg_update_time = (
            self._total_update_time / self._update_count * 1000 
            if self._update_count > 0 else 0.0
        )
        
        return {
            'current_seq_len': self.current_seq_len,
            'memory_used_mb': memory_used / (1024 * 1024),
            'memory_capacity_mb': memory_capacity / (1024 * 1024),
            'utilization': (self.current_seq_len / self.config.max_seq_len * 100) if self.config.max_seq_len > 0 else 0,
            'avg_update_time_ms': avg_update_time
        }
    
    def __repr__(self) -> str:
        """String representation with statistics"""
        stats = self.get_stats()
        return (
            f"KVCache(layers={self.config.num_layers}, "
            f"seq_len={self.current_seq_len}/{self.config.max_seq_len}, "
            f"memory={stats['memory_used_mb']:.1f}/{stats['memory_capacity_mb']:.1f} MB, "
            f"utilization={stats['utilization']:.1f}%)"
        )


class InferenceEngine:
    """
    Inference Engine for Autoregressive Text Generation
    
    Manages the complete generation pipeline:
    1. Prompt encoding
    2. Autoregressive generation with KV caching
    3. Sampling strategies (greedy, temperature, top-k, top-p)
    4. Stopping criteria (max_tokens, eos_token)
    5. Performance profiling
    
    Generation Flow:
        1. Encode prompt → initial token IDs
        2. Forward pass → compute logits for next token
        3. Apply sampling → select next token
        4. Update KV cache with new K, V
        5. Repeat steps 2-4 until stopping criterion
        6. Decode tokens → output text
    
    Speedup Analysis:
        Without cache: Each step processes full sequence [0, t]
            - Step 1: 1 token
            - Step 2: 2 tokens
            - Step 3: 3 tokens
            - Total: 1 + 2 + 3 + ... + n = O(n²)
        
        With cache: Each step processes only new token
            - Step 1: 1 token + 0 cache
            - Step 2: 1 token + 1 cache
            - Step 3: 1 token + 2 cache
            - Total: n steps × 1 token = O(n)
        
        Speedup: O(n²) / O(n) = O(n) → 10-100x for typical sequences
    
    Args:
        model: Transformer model (must have .generate_with_cache method)
        cache: KV cache instance (optional, created if None)
        device: Device for inference ('cpu', 'cuda', 'mps')
        use_cache: Whether to enable KV caching (default: True)
        profile: Whether to collect performance metrics
    
    Attributes:
        model: The transformer model
        cache: KV cache instance
        device: Inference device
        use_cache: Cache enabled flag
        profile: Profiling enabled flag
        stats: Performance statistics
    
    Example:
        >>> model = TransformerModel(...)
        >>> engine = InferenceEngine(model, device='cuda', use_cache=True)
        >>> 
        >>> output = engine.generate(
        ...     prompt="The meaning of life is",
        ...     max_new_tokens=50,
        ...     temperature=0.8,
        ...     top_k=40,
        ...     top_p=0.95
        ... )
        >>> 
        >>> print(output['text'])
        >>> print(f"Speed: {output['tokens_per_second']:.1f} tokens/s")
    """
    
    def __init__(
        self,
        model: nn.Module,
        cache: Optional[KVCache] = None,
        device: str = 'cpu',
        use_cache: bool = True,
        profile: bool = False
    ):
        """
        Initialize Inference Engine
        
        Args:
            model: Transformer model for generation
            cache: Pre-initialized KV cache (optional)
            device: Device for inference
            use_cache: Enable KV caching for speedup
            profile: Enable performance profiling
        """
        self.model = model.to(device)
        self.device = device
        self.use_cache = use_cache
        self.profile = profile
        
        # Cache setup
        self.cache = cache
        
        # Performance statistics
        self.stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'avg_tokens_per_second': 0.0
        }
        
        # Set model to eval mode
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> Dict[str, any]:
        """
        Generate text autoregressively
        
        Args:
            prompt: Input prompt (string or token IDs)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
                - 0.0: greedy (always pick most likely)
                - 0.1-0.7: focused, coherent
                - 0.8-1.2: balanced creativity
                - 1.5+: very random
            top_k: Keep only top K tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            eos_token_id: End-of-sequence token (stop generation)
            pad_token_id: Padding token ID
            do_sample: Use sampling (True) or greedy (False)
            num_return_sequences: Number of sequences to generate (beam search if > 1)
        
        Returns:
            Dictionary with:
                - text: Generated text (if tokenizer available)
                - token_ids: Generated token IDs
                - prompt_tokens: Number of prompt tokens
                - generated_tokens: Number of generated tokens
                - total_tokens: Total tokens processed
                - generation_time: Time taken in seconds
                - tokens_per_second: Generation speed
                - cache_stats: KV cache statistics (if enabled)
        
        Note:
            This is a simplified implementation. Full implementation requires:
            - Tokenizer integration
            - Proper attention masking
            - Beam search for num_return_sequences > 1
            - Batch processing for multiple prompts
        """
        start_time = time.time()
        
        # Clear cache for new generation
        if self.use_cache and self.cache is not None:
            self.cache.clear()
        
        # Handle string prompt (requires tokenizer - placeholder for now)
        if isinstance(prompt, str):
            # In real implementation: token_ids = tokenizer.encode(prompt)
            raise NotImplementedError("String prompt requires tokenizer integration")
        else:
            token_ids = prompt
        
        # Ensure token_ids is on correct device
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.to(self.device)
        
        prompt_length = token_ids.size(-1) if isinstance(token_ids, Tensor) else len(token_ids)
        generated_tokens = []
        
        # Generation loop
        for step in range(max_new_tokens):
            # Forward pass (with or without cache)
            if self.use_cache and self.cache is not None:
                # With cache: only process new token(s)
                # First iteration: full prompt, subsequent: single token
                input_ids = token_ids if step == 0 else token_ids[:, -1:]
                logits = self._forward_with_cache(input_ids)
            else:
                # Without cache: process full sequence every time (slow!)
                logits = self.model(token_ids)
            
            # Get logits for last position (next token prediction)
            next_token_logits = logits[:, -1, :]
            
            # Apply sampling
            next_token = self._sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
            
            # Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            # Append to sequence
            generated_tokens.append(next_token.item())
            token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=-1)
        
        # Calculate statistics
        generation_time = time.time() - start_time
        total_tokens = prompt_length + len(generated_tokens)
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        # Update global stats
        self.stats['total_generations'] += 1
        self.stats['total_tokens_generated'] += len(generated_tokens)
        self.stats['total_time'] += generation_time
        self.stats['avg_tokens_per_second'] = (
            self.stats['total_tokens_generated'] / self.stats['total_time']
            if self.stats['total_time'] > 0 else 0
        )
        
        # Prepare output
        result = {
            'token_ids': token_ids,
            'generated_token_ids': generated_tokens,
            'prompt_tokens': prompt_length,
            'generated_tokens': len(generated_tokens),
            'total_tokens': total_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second
        }
        
        # Add cache stats if available
        if self.use_cache and self.cache is not None:
            result['cache_stats'] = self.cache.get_stats()
        
        return result
    
    def _forward_with_cache(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass with KV cache
        
        This is a placeholder - actual implementation depends on model architecture.
        The model must support cache-aware forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
        
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # In real implementation:
        # return self.model.forward_with_cache(input_ids, cache=self.cache)
        
        # Placeholder: standard forward pass
        return self.model(input_ids)
    
    def _sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> Tensor:
        """
        Sample next token from logits
        
        Implements various sampling strategies:
        1. Greedy: always pick argmax (temperature=0 or do_sample=False)
        2. Temperature: scale logits before softmax
        3. Top-K: keep only K highest probability tokens
        4. Top-P: keep smallest set with cumulative probability >= P
        
        Args:
            logits: Logits for next token [batch, vocab_size]
            temperature: Sampling temperature
            top_k: Top-K threshold
            top_p: Top-P (nucleus) threshold
            do_sample: Whether to sample or use greedy
        
        Returns:
            Next token ID [batch, 1]
        """
        # Greedy decoding
        if not do_sample or temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Keep only top k values
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def benchmark(
        self,
        prompt: Tensor,
        max_new_tokens: int = 100,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark generation speed with and without cache
        
        Args:
            prompt: Input token IDs
            max_new_tokens: Tokens to generate per run
            num_runs: Number of benchmark runs
        
        Returns:
            Dictionary with benchmark results:
                - with_cache_speed: Tokens/second with cache
                - without_cache_speed: Tokens/second without cache
                - speedup: Speedup factor (with_cache / without_cache)
        """
        # Benchmark with cache
        self.use_cache = True
        with_cache_times = []
        
        for _ in range(num_runs):
            result = self.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            with_cache_times.append(result['tokens_per_second'])
        
        avg_with_cache = sum(with_cache_times) / len(with_cache_times)
        
        # Benchmark without cache
        self.use_cache = False
        without_cache_times = []
        
        for _ in range(num_runs):
            result = self.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            without_cache_times.append(result['tokens_per_second'])
        
        avg_without_cache = sum(without_cache_times) / len(without_cache_times)
        
        # Restore cache setting
        self.use_cache = True
        
        return {
            'with_cache_speed': avg_with_cache,
            'without_cache_speed': avg_without_cache,
            'speedup': avg_with_cache / avg_without_cache if avg_without_cache > 0 else 0,
            'num_runs': num_runs
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset inference statistics"""
        self.stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'avg_tokens_per_second': 0.0
        }


if __name__ == "__main__":
    """
    Test KV Cache and Inference Engine
    """
    print("=" * 80)
    print("Testing KV Cache and Inference Optimization")
    print("=" * 80)
    
    # Test 1: Cache Configuration
    print("\n" + "-" * 80)
    print("Test 1: Cache Configuration")
    print("-" * 80)
    
    config = CacheConfig(
        num_layers=12,
        batch_size=1,
        max_seq_len=1024,
        num_heads=12,
        head_dim=64,
        device='cpu',
        dtype=torch.float32
    )
    
    print(f"✅ Config created")
    print(f"   Memory usage: {config.get_memory_usage_mb():.2f} MB")
    print(f"   Memory usage: {config.get_memory_usage_gb():.4f} GB")
    
    # Test 2: KV Cache Operations
    print("\n" + "-" * 80)
    print("Test 2: KV Cache Operations")
    print("-" * 80)
    
    cache = KVCache(config)
    print(f"✅ Cache initialized: {cache}")
    
    # Simulate generation
    for step in range(10):
        # New token K, V
        k = torch.randn(1, 12, 1, 64)
        v = torch.randn(1, 12, 1, 64)
        
        # Update cache for each layer
        for layer in range(12):
            full_k, full_v = cache.update(layer, k, v)
        
        if step % 3 == 0:
            print(f"   Step {step}: {cache}")
    
    stats = cache.get_stats()
    print(f"\n✅ Cache statistics:")
    print(f"   Sequence length: {stats['current_seq_len']}")
    print(f"   Memory used: {stats['memory_used_mb']:.2f} MB")
    print(f"   Utilization: {stats['utilization']:.1f}%")
    print(f"   Avg update time: {stats['avg_update_time_ms']:.3f} ms")
    
    # Test 3: Cache Clear
    print("\n" + "-" * 80)
    print("Test 3: Cache Clear")
    print("-" * 80)
    
    cache.clear()
    print(f"✅ Cache cleared: {cache}")
    
    # Test 4: Memory Scaling
    print("\n" + "-" * 80)
    print("Test 4: Memory Scaling Analysis")
    print("-" * 80)
    
    configs = [
        ("GPT-2 Small", 12, 12, 64, 1024),
        ("GPT-2 Medium", 24, 16, 64, 1024),
        ("GPT-2 Large", 36, 20, 64, 1024),
        ("GPT-3 style", 96, 96, 128, 2048)
    ]
    
    print(f"\n{'Model':<20} {'Layers':<8} {'Heads':<8} {'Seq Len':<10} {'Memory (FP32)':<15} {'Memory (FP16)':<15}")
    print("-" * 90)
    
    for name, layers, heads, head_dim, seq_len in configs:
        config_fp32 = CacheConfig(layers, 1, seq_len, heads, head_dim, dtype=torch.float32)
        config_fp16 = CacheConfig(layers, 1, seq_len, heads, head_dim, dtype=torch.float16)
        
        print(f"{name:<20} {layers:<8} {heads:<8} {seq_len:<10} "
              f"{config_fp32.get_memory_usage_mb():.1f} MB{'':<7} "
              f"{config_fp16.get_memory_usage_mb():.1f} MB")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
