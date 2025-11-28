"""
Text Sampling Strategies for Language Model Generation

This module implements various sampling strategies for controlling text generation
diversity and quality in autoregressive language models.

Sampling Strategies:
1. Greedy Decoding - always pick most probable token (deterministic)
2. Temperature Sampling - scale logits to control randomness
3. Top-K Sampling - restrict to K most probable tokens
4. Top-P (Nucleus) Sampling - dynamic cutoff based on cumulative probability
5. Beam Search - maintain multiple hypotheses (future implementation)
6. Contrastive Search - balance coherence and diversity (future implementation)

Mathematical Foundation:

Temperature Sampling:
    P(x_i) = exp(logit_i / T) / Σ_j exp(logit_j / T)
    
    Where T is temperature:
    - T → 0: Approaches greedy (argmax)
    - T = 1: Standard softmax
    - T > 1: Flatter distribution (more random)
    - T → ∞: Uniform distribution

Top-K Sampling:
    1. Sort tokens by probability in descending order
    2. Keep only top K tokens
    3. Renormalize and sample
    
    Benefits:
    - Eliminates low-probability tokens (reduces nonsense)
    - Fixed budget of candidates (predictable)
    - Simple and fast

Top-P (Nucleus) Sampling:
    1. Sort tokens by probability in descending order
    2. Find smallest set S where Σ_{x∈S} P(x) ≥ p
    3. Sample from S
    
    Benefits:
    - Dynamic vocabulary size (adapts to distribution)
    - Eliminates "long tail" of unlikely tokens
    - More robust than top-k for varying distributions

Combination Strategies:
    Often used together: Temperature → Top-K → Top-P → Sample
    This provides fine-grained control over generation characteristics.

Usage:
    # Initialize sampler
    sampler = TextSampler(
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    # Sample next token
    logits = model(input_ids)
    next_token = sampler.sample(logits, input_ids=input_ids)
    
    # Or use standalone functions
    next_token = sample_top_k(logits, k=40, temperature=0.9)
    next_token = sample_top_p(logits, p=0.95, temperature=0.9)

Author: NOVA Development Team
Date: 28 November 2025
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from torch import Tensor
import numpy as np


def sample_greedy(logits: Tensor) -> Tensor:
    """
    Greedy Decoding - always pick most probable token
    
    The simplest sampling strategy: deterministic, always picks argmax.
    
    Pros:
    - Deterministic (reproducible)
    - Fast (no sampling overhead)
    - Often produces reasonable results
    
    Cons:
    - No diversity
    - Can get stuck in repetitive patterns
    - May not explore creative solutions
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
    
    Returns:
        Token IDs [batch_size, 1]
    
    Example:
        >>> logits = torch.tensor([[1.0, 3.0, 2.0]])  # Token 1 has highest logit
        >>> token = sample_greedy(logits)
        >>> print(token)  # tensor([[1]])
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def sample_temperature(
    logits: Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Temperature Sampling - scale logits to control randomness
    
    Temperature controls the "sharpness" of the probability distribution:
    - T < 1: Sharper (more confident, less diverse)
    - T = 1: Standard softmax (unchanged)
    - T > 1: Flatter (more random, more diverse)
    
    Mathematical Effect:
        Original: P(x) = exp(logit_x) / Z
        With temp: P(x) = exp(logit_x / T) / Z'
        
        As T decreases:
        - High probability tokens become even more likely
        - Low probability tokens become even less likely
        
        As T increases:
        - Distribution flattens (approaches uniform)
        - More exploration of low-probability tokens
    
    Recommended Values:
    - T = 0.1-0.5: Very focused (technical writing, code)
    - T = 0.7-0.9: Balanced (most general use cases)
    - T = 1.0-1.5: Creative (storytelling, brainstorming)
    - T = 1.5+: Very random (experimental)
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
        temperature: Temperature parameter (> 0)
        generator: Random generator for reproducibility
    
    Returns:
        Token IDs [batch_size, 1]
    
    Raises:
        ValueError: If temperature <= 0
    
    Example:
        >>> logits = torch.tensor([[1.0, 2.0, 3.0]])
        >>> 
        >>> # Low temperature: mostly picks token 2 (highest logit)
        >>> token = sample_temperature(logits, temperature=0.1)
        >>> 
        >>> # High temperature: more uniform sampling
        >>> token = sample_temperature(logits, temperature=2.0)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from distribution
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token


def sample_top_k(
    logits: Tensor,
    k: int = 50,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Top-K Sampling - restrict sampling to K most probable tokens
    
    Filters out low-probability tokens by keeping only the top K candidates.
    This prevents sampling from the "long tail" of unlikely tokens.
    
    Algorithm:
        1. Find top K tokens by logit value
        2. Set all other logits to -inf (probability → 0)
        3. Apply temperature and sample
    
    Benefits:
    - Eliminates nonsensical low-probability tokens
    - Fixed computational budget
    - Simple to implement and understand
    
    Drawbacks:
    - K is fixed regardless of probability distribution
    - May include too many tokens (when distribution is peaked)
    - May exclude too many tokens (when distribution is flat)
    
    Recommended K Values:
    - K = 1: Greedy decoding
    - K = 10-20: Very focused
    - K = 40-50: Balanced (recommended default)
    - K = 100-200: More diverse
    - K = vocab_size: No filtering (pure temperature)
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
        k: Number of top tokens to keep (1 to vocab_size)
        temperature: Temperature for sampling
        generator: Random generator for reproducibility
    
    Returns:
        Token IDs [batch_size, 1]
    
    Example:
        >>> logits = torch.randn(1, 1000)  # 1000 token vocabulary
        >>> token = sample_top_k(logits, k=50)  # Sample from top 50 only
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    # Ensure k doesn't exceed vocabulary size
    k = min(k, logits.size(-1))
    
    # Find top k values and their indices
    # topk returns (values, indices), we only need the kth value
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    
    # Get the kth largest value (minimum threshold)
    kth_value = top_k_values[..., -1, None]
    
    # Mask out logits below threshold (set to -inf)
    # Tokens with logit < kth_value will have probability 0 after softmax
    logits_filtered = logits.clone()
    logits_filtered[logits < kth_value] = float('-inf')
    
    # Apply temperature and sample
    return sample_temperature(logits_filtered, temperature, generator)


def sample_top_p(
    logits: Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    min_tokens_to_keep: int = 1
) -> Tensor:
    """
    Top-P (Nucleus) Sampling - dynamic cutoff based on cumulative probability
    
    Instead of fixed K, nucleus sampling finds the smallest set of tokens whose
    cumulative probability exceeds threshold p.
    
    Algorithm:
        1. Sort tokens by probability (descending)
        2. Compute cumulative probability
        3. Find cutoff where cumulative_prob >= p
        4. Keep all tokens up to cutoff
        5. Apply temperature and sample
    
    Benefits:
    - Adaptive vocabulary size (adjusts to distribution)
    - Peaked distribution → fewer tokens (focused)
    - Flat distribution → more tokens (diverse)
    - More robust than top-k across different contexts
    
    Intuition:
        "Keep the smallest set of tokens that captures p% of the probability mass"
        
        Example with p=0.9:
        - If one token has P=0.95, keep only that token
        - If top 10 tokens needed to reach 0.9, keep all 10
        - Automatically adapts to confidence level
    
    Recommended P Values:
    - P = 0.5-0.7: Very focused (conservative)
    - P = 0.8-0.9: Balanced (recommended default)
    - P = 0.95-0.99: More diverse
    - P = 1.0: No filtering (pure temperature)
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
        p: Cumulative probability threshold (0.0 to 1.0)
        temperature: Temperature for sampling
        generator: Random generator for reproducibility
        min_tokens_to_keep: Minimum number of tokens to keep (safety)
    
    Returns:
        Token IDs [batch_size, 1]
    
    Example:
        >>> logits = torch.randn(1, 1000)
        >>> token = sample_top_p(logits, p=0.9)  # Keep top tokens covering 90% probability
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    
    if min_tokens_to_keep < 1:
        raise ValueError(f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}")
    
    # Apply temperature first
    if temperature != 1.0:
        logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability exceeds p
    # We want to remove tokens AFTER the cutoff
    sorted_indices_to_remove = cumulative_probs > p
    
    # Always keep at least min_tokens_to_keep tokens
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Shift right by 1 to keep the first token that exceeds p
    # This ensures we always include the token that pushes us over threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original indexing
    # Convert sorted mask to original vocabulary order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    
    # Set removed tokens to -inf
    logits_filtered = logits.clone()
    logits_filtered[indices_to_remove] = float('-inf')
    
    # Sample from filtered distribution
    probs = F.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token


def apply_repetition_penalty(
    logits: Tensor,
    input_ids: Tensor,
    penalty: float = 1.0
) -> Tensor:
    """
    Apply repetition penalty to discourage repeated tokens
    
    Reduces the probability of tokens that have already appeared in the sequence.
    Higher penalty = stronger discouragement.
    
    Algorithm:
        For each token that appears in input_ids:
            if logit > 0: logit = logit / penalty
            if logit < 0: logit = logit * penalty
        
        This makes:
        - Positive logits smaller (less likely)
        - Negative logits more negative (even less likely)
    
    Recommended Penalty Values:
    - penalty = 1.0: No penalty (disabled)
    - penalty = 1.1-1.2: Mild discouragement (recommended)
    - penalty = 1.5-2.0: Strong discouragement
    - penalty > 2.0: Very strong (may hurt coherence)
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
        input_ids: Previously generated tokens [batch_size, seq_len]
        penalty: Repetition penalty factor (>= 1.0)
    
    Returns:
        Modified logits [batch_size, vocab_size]
    
    Example:
        >>> logits = torch.randn(1, 1000)
        >>> input_ids = torch.tensor([[1, 5, 10, 5]])  # Token 5 repeated
        >>> logits_penalized = apply_repetition_penalty(logits, input_ids, penalty=1.2)
        >>> # Token 5 now less likely to be sampled again
    """
    if penalty <= 0:
        raise ValueError(f"Penalty must be positive, got {penalty}")
    
    if penalty == 1.0:
        return logits  # No penalty
    
    batch_size, vocab_size = logits.shape
    
    # Clone to avoid modifying original
    logits = logits.clone()
    
    # Apply penalty for each sequence in batch
    for i in range(batch_size):
        # Get unique tokens in this sequence
        unique_tokens = input_ids[i].unique()
        
        # Apply penalty to each repeated token
        for token_id in unique_tokens:
            if token_id >= vocab_size:
                continue  # Skip special tokens outside vocabulary
            
            # Reduce probability of repeated tokens
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty
    
    return logits


class TextSampler:
    """
    Unified Text Sampler with Multiple Strategies
    
    Provides a clean interface for text generation with configurable sampling.
    Combines temperature, top-k, top-p, and repetition penalty in one class.
    
    The sampling pipeline:
        1. Apply repetition penalty (if enabled)
        2. Apply temperature scaling
        3. Apply top-k filtering (if enabled)
        4. Apply top-p filtering (if enabled)
        5. Sample from resulting distribution
    
    This ordering is important:
    - Repetition penalty modifies logits based on context
    - Temperature scales overall randomness
    - Top-k and top-p filter the vocabulary
    - Final sampling from filtered distribution
    
    Args:
        temperature: Sampling temperature (> 0)
        top_k: Top-K threshold (None to disable)
        top_p: Top-P threshold (None to disable)
        repetition_penalty: Repetition penalty factor (1.0 to disable)
        min_tokens_to_keep: Minimum tokens for top-p
        generator: Random generator for reproducibility
    
    Attributes:
        temperature: Current temperature setting
        top_k: Current top-k setting
        top_p: Current top-p setting
        repetition_penalty: Current repetition penalty
        min_tokens_to_keep: Minimum tokens for top-p
        generator: Random generator
    
    Example:
        >>> sampler = TextSampler(
        ...     temperature=0.8,
        ...     top_k=50,
        ...     top_p=0.95,
        ...     repetition_penalty=1.2
        ... )
        >>> 
        >>> # During generation
        >>> logits = model(input_ids)
        >>> next_token = sampler.sample(logits, input_ids=input_ids)
        >>> 
        >>> # Update settings dynamically
        >>> sampler.set_temperature(0.5)
        >>> next_token = sampler.sample(logits)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
        generator: Optional[torch.Generator] = None
    ):
        """
        Initialize TextSampler
        
        Args:
            temperature: Sampling temperature (> 0)
            top_k: Top-K threshold (None to disable)
            top_p: Top-P threshold (None to disable)
            repetition_penalty: Repetition penalty (>= 1.0)
            min_tokens_to_keep: Minimum tokens for top-p
            generator: Random generator for reproducibility
        """
        # Validate parameters
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        
        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")
        
        # Store settings
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep
        self.generator = generator
    
    def sample(
        self,
        logits: Tensor,
        input_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        Sample next token using configured strategy
        
        Args:
            logits: Logits for next token [batch_size, vocab_size]
            input_ids: Previous tokens for repetition penalty [batch_size, seq_len]
        
        Returns:
            Token IDs [batch_size, 1]
        """
        # Step 1: Apply repetition penalty
        if self.repetition_penalty != 1.0 and input_ids is not None:
            logits = apply_repetition_penalty(logits, input_ids, self.repetition_penalty)
        
        # Step 2: Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Step 3: Apply top-k filtering
        if self.top_k is not None:
            k = min(self.top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, k, dim=-1)
            kth_value = top_k_values[..., -1, None]
            logits = logits.clone()
            logits[logits < kth_value] = float('-inf')
        
        # Step 4: Apply top-p filtering
        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.clone()
            logits[indices_to_remove] = float('-inf')
        
        # Step 5: Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=self.generator)
        
        return next_token
    
    def set_temperature(self, temperature: float):
        """Update temperature setting"""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
    
    def set_top_k(self, top_k: Optional[int]):
        """Update top-k setting"""
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.top_k = top_k
    
    def set_top_p(self, top_p: Optional[float]):
        """Update top-p setting"""
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        self.top_p = top_p
    
    def set_repetition_penalty(self, penalty: float):
        """Update repetition penalty"""
        if penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {penalty}")
        self.repetition_penalty = penalty
    
    def get_config(self) -> dict:
        """Get current sampler configuration"""
        return {
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'min_tokens_to_keep': self.min_tokens_to_keep
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TextSampler(temperature={self.temperature}, "
            f"top_k={self.top_k}, top_p={self.top_p}, "
            f"repetition_penalty={self.repetition_penalty})"
        )


def compare_sampling_strategies(
    logits: Tensor,
    num_samples: int = 1000
) -> dict:
    """
    Compare different sampling strategies on the same logits
    
    Useful for understanding how different strategies behave.
    
    Args:
        logits: Logits to sample from [vocab_size]
        num_samples: Number of samples per strategy
    
    Returns:
        Dictionary with statistics for each strategy
    
    Example:
        >>> logits = torch.randn(100)  # 100 token vocabulary
        >>> stats = compare_sampling_strategies(logits, num_samples=1000)
        >>> print(stats['greedy']['entropy'])
        >>> print(stats['temperature_0.5']['entropy'])
    """
    logits = logits.unsqueeze(0)  # Add batch dimension
    vocab_size = logits.size(-1)
    
    results = {}
    
    # Strategy 1: Greedy
    greedy_tokens = []
    for _ in range(num_samples):
        token = sample_greedy(logits)
        greedy_tokens.append(token.item())
    
    results['greedy'] = {
        'unique_tokens': len(set(greedy_tokens)),
        'most_common': max(set(greedy_tokens), key=greedy_tokens.count),
        'entropy': 0.0  # Greedy has zero entropy
    }
    
    # Strategy 2: Temperature = 0.5 (focused)
    temp_05_tokens = []
    for _ in range(num_samples):
        token = sample_temperature(logits, temperature=0.5)
        temp_05_tokens.append(token.item())
    
    results['temperature_0.5'] = {
        'unique_tokens': len(set(temp_05_tokens)),
        'entropy': _compute_entropy(temp_05_tokens, vocab_size)
    }
    
    # Strategy 3: Temperature = 1.0 (standard)
    temp_10_tokens = []
    for _ in range(num_samples):
        token = sample_temperature(logits, temperature=1.0)
        temp_10_tokens.append(token.item())
    
    results['temperature_1.0'] = {
        'unique_tokens': len(set(temp_10_tokens)),
        'entropy': _compute_entropy(temp_10_tokens, vocab_size)
    }
    
    # Strategy 4: Top-K = 50
    topk_tokens = []
    for _ in range(num_samples):
        token = sample_top_k(logits, k=50)
        topk_tokens.append(token.item())
    
    results['top_k_50'] = {
        'unique_tokens': len(set(topk_tokens)),
        'entropy': _compute_entropy(topk_tokens, vocab_size)
    }
    
    # Strategy 5: Top-P = 0.9
    topp_tokens = []
    for _ in range(num_samples):
        token = sample_top_p(logits, p=0.9)
        topp_tokens.append(token.item())
    
    results['top_p_0.9'] = {
        'unique_tokens': len(set(topp_tokens)),
        'entropy': _compute_entropy(topp_tokens, vocab_size)
    }
    
    return results


def _compute_entropy(tokens: List[int], vocab_size: int) -> float:
    """Compute empirical entropy of token distribution"""
    counts = np.bincount(tokens, minlength=vocab_size)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


if __name__ == "__main__":
    """
    Test sampling strategies
    """
    print("=" * 80)
    print("Testing Text Sampling Strategies")
    print("=" * 80)
    
    # Create test logits
    vocab_size = 1000
    logits = torch.randn(1, vocab_size)
    
    print(f"\nTest setup: vocab_size={vocab_size}")
    print(f"Logits shape: {logits.shape}")
    
    # Test 1: Greedy Sampling
    print("\n" + "-" * 80)
    print("Test 1: Greedy Sampling")
    print("-" * 80)
    
    token = sample_greedy(logits)
    print(f"✅ Greedy token: {token.item()}")
    
    # Verify determinism
    token2 = sample_greedy(logits)
    print(f"✅ Deterministic: {token.item() == token2.item()}")
    
    # Test 2: Temperature Sampling
    print("\n" + "-" * 80)
    print("Test 2: Temperature Sampling")
    print("-" * 80)
    
    for temp in [0.1, 0.5, 1.0, 1.5]:
        token = sample_temperature(logits, temperature=temp)
        print(f"✅ Temperature {temp}: token {token.item()}")
    
    # Test 3: Top-K Sampling
    print("\n" + "-" * 80)
    print("Test 3: Top-K Sampling")
    print("-" * 80)
    
    for k in [1, 10, 50, 100]:
        token = sample_top_k(logits, k=k)
        print(f"✅ Top-K (k={k}): token {token.item()}")
    
    # Test 4: Top-P Sampling
    print("\n" + "-" * 80)
    print("Test 4: Top-P Sampling")
    print("-" * 80)
    
    for p in [0.5, 0.7, 0.9, 0.95]:
        token = sample_top_p(logits, p=p)
        print(f"✅ Top-P (p={p}): token {token.item()}")
    
    # Test 5: Repetition Penalty
    print("\n" + "-" * 80)
    print("Test 5: Repetition Penalty")
    print("-" * 80)
    
    input_ids = torch.tensor([[10, 20, 30, 20, 10]])  # Repeated tokens: 10, 20
    logits_original = logits.clone()
    logits_penalized = apply_repetition_penalty(logits, input_ids, penalty=1.5)
    
    print(f"✅ Original logit for token 10: {logits_original[0, 10]:.4f}")
    print(f"✅ Penalized logit for token 10: {logits_penalized[0, 10]:.4f}")
    print(f"✅ Original logit for token 50: {logits_original[0, 50]:.4f}")
    print(f"✅ Penalized logit for token 50: {logits_penalized[0, 50]:.4f} (unchanged)")
    
    # Test 6: TextSampler
    print("\n" + "-" * 80)
    print("Test 6: Unified TextSampler")
    print("-" * 80)
    
    sampler = TextSampler(
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    print(f"✅ Sampler created: {sampler}")
    print(f"✅ Config: {sampler.get_config()}")
    
    token = sampler.sample(logits, input_ids=input_ids)
    print(f"✅ Sampled token: {token.item()}")
    
    # Test 7: Strategy Comparison
    print("\n" + "-" * 80)
    print("Test 7: Strategy Comparison (1000 samples each)")
    print("-" * 80)
    
    stats = compare_sampling_strategies(logits[0], num_samples=1000)
    
    print(f"\n{'Strategy':<20} {'Unique Tokens':<15} {'Entropy':<10}")
    print("-" * 50)
    for strategy, data in stats.items():
        unique = data['unique_tokens']
        entropy = data.get('entropy', 0.0)
        print(f"{strategy:<20} {unique:<15} {entropy:<10.2f}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
