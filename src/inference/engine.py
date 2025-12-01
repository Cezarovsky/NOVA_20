"""
Inference Engine Core

Efficient text generation with various strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Generation strategy
    strategy: str = 'greedy'  # greedy, beam_search, sampling, top_k, top_p, nucleus
    
    # Common parameters
    max_length: int = 512
    min_length: int = 1
    temperature: float = 1.0
    
    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    
    # Sampling
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Control
    do_sample: bool = False
    num_return_sequences: int = 1


class InferenceEngine:
    """
    High-performance inference engine.
    
    Supports:
    - Multiple generation strategies
    - KV-cache for efficiency
    - Batch inference
    - Streaming generation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_kv_cache: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Language model
            device: Device to run inference on
            use_kv_cache: Use KV-cache for efficiency
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_kv_cache = use_kv_cache
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """
        Generate text from input.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            config: Generation configuration
            
        Returns:
            Generated token IDs [batch_size, new_seq_len]
        """
        if config is None:
            config = GenerationConfig()
        
        # Move to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Select generation strategy
        if config.strategy == 'greedy':
            return self._greedy_generate(input_ids, attention_mask, config)
        elif config.strategy == 'beam_search':
            return self._beam_search_generate(input_ids, attention_mask, config)
        elif config.strategy in ['sampling', 'top_k', 'top_p', 'nucleus']:
            return self._sampling_generate(input_ids, attention_mask, config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Greedy decoding."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(config.max_length - seq_len):
            # Forward pass
            outputs = self.model(generated, attention_mask=attention_mask)
            
            # Get logits for next token
            next_token_logits = outputs[:, -1, :] / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )
            
            # Greedy selection
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)
            
            # Check for EOS
            if (next_tokens == config.eos_token_id).all():
                break
        
        return generated
    
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Beam search decoding."""
        batch_size, seq_len = input_ids.shape
        num_beams = config.num_beams
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, seq_len)
        input_ids = input_ids.reshape(batch_size * num_beams, seq_len)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, seq_len)
            attention_mask = attention_mask.reshape(batch_size * num_beams, seq_len)
        
        # Beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        
        generated = input_ids.clone()
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(config.max_length - seq_len):
            # Forward pass
            outputs = self.model(generated, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :] / config.temperature
            
            # Get scores
            vocab_size = next_token_logits.shape[-1]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores.reshape(batch_size, num_beams, vocab_size)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            next_token_scores = next_token_scores.reshape(batch_size, num_beams * vocab_size)
            
            # Select top beams
            top_scores, top_indices = torch.topk(next_token_scores, num_beams, dim=-1)
            
            # Get beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            beam_scores = top_scores
            
            # Gather beams
            beam_idx = beam_indices + torch.arange(batch_size, device=self.device).unsqueeze(1) * num_beams
            beam_idx = beam_idx.reshape(-1)
            generated = generated[beam_idx]
            
            # Append tokens
            next_tokens = token_indices.reshape(-1, 1)
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = attention_mask[beam_idx]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size * num_beams, 1), device=self.device)
                ], dim=1)
            
            # Check for EOS
            eos_mask = next_tokens.squeeze(-1) == config.eos_token_id
            done = done | eos_mask.reshape(batch_size, num_beams).all(dim=1)
            
            if config.early_stopping and done.all():
                break
        
        # Return best beam for each batch
        generated = generated.reshape(batch_size, num_beams, -1)
        return generated[:, 0, :]  # Best beam
    
    def _sampling_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Sampling-based generation (top-k, top-p, nucleus)."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(config.max_length - seq_len):
            # Forward pass
            outputs = self.model(generated, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :] / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)
            
            # Check for EOS
            if (next_tokens == config.eos_token_id).all():
                break
        
        return generated
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for batch_idx in range(logits.shape[0]):
            for token in set(generated[batch_idx].tolist()):
                if logits[batch_idx, token] < 0:
                    logits[batch_idx, token] *= penalty
                else:
                    logits[batch_idx, token] /= penalty
        
        return logits
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Filter top-k tokens."""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """Filter top-p (nucleus) tokens."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        
        return logits


class BatchInference:
    """
    Efficient batch inference.
    
    Processes multiple inputs simultaneously with padding.
    """
    
    def __init__(
        self,
        engine: InferenceEngine,
        batch_size: int = 32,
        pad_token_id: int = 0,
    ):
        """
        Initialize batch inference.
        
        Args:
            engine: Inference engine
            batch_size: Maximum batch size
            pad_token_id: Padding token ID
        """
        self.engine = engine
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
    
    def generate(
        self,
        input_ids_list: List[torch.Tensor],
        config: Optional[GenerationConfig] = None,
    ) -> List[torch.Tensor]:
        """
        Generate for multiple inputs.
        
        Args:
            input_ids_list: List of input token IDs
            config: Generation configuration
            
        Returns:
            List of generated token IDs
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(input_ids_list), self.batch_size):
            batch = input_ids_list[i:i + self.batch_size]
            
            # Pad batch
            max_len = max(ids.shape[0] for ids in batch)
            padded_batch = []
            attention_masks = []
            
            for ids in batch:
                padding_len = max_len - ids.shape[0]
                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_len,), self.pad_token_id, dtype=ids.dtype)
                ])
                mask = torch.cat([
                    torch.ones(ids.shape[0], dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long)
                ])
                
                padded_batch.append(padded_ids)
                attention_masks.append(mask)
            
            # Stack into batch
            input_ids = torch.stack(padded_batch)
            attention_mask = torch.stack(attention_masks)
            
            # Generate
            outputs = self.engine.generate(input_ids, attention_mask, config)
            
            # Unpack results
            for output in outputs:
                all_results.append(output)
        
        return all_results


class StreamingInference:
    """
    Streaming inference for real-time generation.
    
    Yields tokens as they are generated.
    """
    
    def __init__(self, engine: InferenceEngine):
        """
        Initialize streaming inference.
        
        Args:
            engine: Inference engine
        """
        self.engine = engine
    
    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Generate tokens in streaming fashion.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            config: Generation configuration
            
        Yields:
            Generated token IDs (one at a time)
        """
        if config is None:
            config = GenerationConfig()
        
        # Move to device
        input_ids = input_ids.to(self.engine.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.engine.device)
        
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(config.max_length - seq_len):
            # Forward pass
            outputs = self.engine.model(generated, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :] / config.temperature
            
            # Select next token (greedy for streaming)
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Yield token
            yield next_tokens
            
            # Append to sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.engine.device)
                ], dim=1)
            
            # Check for EOS
            if (next_tokens == config.eos_token_id).all():
                break
