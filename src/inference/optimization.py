"""
Inference Optimization

Techniques to speed up inference:
- KV-cache
- Quantization
- Pruning
- Knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import copy


@dataclass
class OptimizationConfig:
    """Configuration for inference optimization."""
    
    # KV-cache
    use_kv_cache: bool = True
    
    # Quantization
    quantize: bool = False
    quantization_bits: int = 8  # 8-bit or 4-bit
    quantization_method: str = 'dynamic'  # dynamic, static
    
    # Pruning
    prune: bool = False
    pruning_ratio: float = 0.3  # 30% pruning
    pruning_method: str = 'magnitude'  # magnitude, structured
    
    # Distillation
    distill: bool = False
    teacher_model: Optional[nn.Module] = None
    temperature: float = 2.0
    alpha: float = 0.5  # Balance between hard and soft targets


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    
    Caches attention keys and values to avoid recomputation.
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_length: int = 2048,
        device: str = 'cuda',
    ):
        """
        Initialize KV cache.
        
        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_length: Maximum sequence length
            device: Device to store cache
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.device = device
        
        # Initialize cache
        self.key_cache = []
        self.value_cache = []
        
        for _ in range(num_layers):
            key = torch.zeros(
                batch_size, num_heads, max_length, head_dim,
                device=device, dtype=torch.float32
            )
            value = torch.zeros(
                batch_size, num_heads, max_length, head_dim,
                device=device, dtype=torch.float32
            )
            self.key_cache.append(key)
            self.value_cache.append(value)
        
        self.current_length = 0
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer.
        
        Args:
            layer_idx: Layer index
            key: New keys [batch_size, num_heads, seq_len, head_dim]
            value: New values [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Updated keys and values
        """
        seq_len = key.shape[2]
        
        # Update cache
        self.key_cache[layer_idx][:, :, self.current_length:self.current_length + seq_len] = key
        self.value_cache[layer_idx][:, :, self.current_length:self.current_length + seq_len] = value
        
        # Return full cached keys and values
        return (
            self.key_cache[layer_idx][:, :, :self.current_length + seq_len],
            self.value_cache[layer_idx][:, :, :self.current_length + seq_len]
        )
    
    def increment_length(self, length: int = 1):
        """Increment current length."""
        self.current_length += length
    
    def reset(self):
        """Reset cache."""
        self.current_length = 0
    
    def clear(self):
        """Clear cache completely."""
        self.key_cache = []
        self.value_cache = []
        self.current_length = 0


class QuantizedModel:
    """
    Quantized model for faster inference.
    
    Reduces precision (8-bit or 4-bit) to speed up computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        method: str = 'dynamic',
    ):
        """
        Initialize quantized model.
        
        Args:
            model: Original model
            bits: Number of bits (8 or 4)
            method: Quantization method (dynamic or static)
        """
        self.model = model
        self.bits = bits
        self.method = method
        
        # Quantize model
        self.quantized_model = self._quantize()
    
    def _quantize(self) -> nn.Module:
        """Quantize model."""
        if self.method == 'dynamic':
            # Dynamic quantization (for inference only)
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8 if self.bits == 8 else torch.quint8
            )
        elif self.method == 'static':
            # Static quantization (requires calibration)
            model = copy.deepcopy(self.model)
            model.eval()
            
            # Prepare for quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibration would happen here with representative data
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
        
        return quantized_model
    
    def __call__(self, *args, **kwargs):
        """Forward pass through quantized model."""
        return self.quantized_model(*args, **kwargs)
    
    def save(self, path: str):
        """Save quantized model."""
        torch.save(self.quantized_model.state_dict(), path)
    
    def load(self, path: str):
        """Load quantized model."""
        self.quantized_model.load_state_dict(torch.load(path))


class PrunedModel:
    """
    Pruned model for faster inference.
    
    Removes less important weights/neurons.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = 'magnitude',
    ):
        """
        Initialize pruned model.
        
        Args:
            model: Original model
            pruning_ratio: Ratio of weights to prune
            method: Pruning method (magnitude or structured)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.method = method
        
        # Prune model
        self._prune()
    
    def _prune(self):
        """Prune model weights."""
        import torch.nn.utils.prune as prune
        
        if self.method == 'magnitude':
            # Magnitude-based pruning (unstructured)
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(
                        module, name='weight', amount=self.pruning_ratio
                    )
        
        elif self.method == 'structured':
            # Structured pruning (removes entire neurons)
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module, name='weight', amount=self.pruning_ratio, n=2, dim=0
                    )
        
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")
    
    def make_permanent(self):
        """Make pruning permanent (remove masks)."""
        import torch.nn.utils.prune as prune
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
    
    def __call__(self, *args, **kwargs):
        """Forward pass through pruned model."""
        return self.model(*args, **kwargs)


class DistilledModel:
    """
    Distilled model (student) trained from teacher.
    
    Smaller model that mimics larger teacher model.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distilled model.
        
        Args:
            student_model: Smaller student model
            teacher_model: Larger teacher model
            temperature: Distillation temperature
            alpha: Balance between hard and soft targets
        """
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Combines:
        - Soft targets (from teacher)
        - Hard targets (true labels)
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: True labels
            
        Returns:
            Distillation loss
        """
        # Soft target loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard target loss (cross entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step.
        
        Args:
            inputs: Input data
            labels: True labels
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        # Student forward pass
        student_logits = self.student(inputs)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Compute loss
        loss = self.compute_distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def __call__(self, *args, **kwargs):
        """Forward pass through student model."""
        return self.student(*args, **kwargs)


class CachedAttention(nn.Module):
    """
    Multi-head attention with KV-cache support.
    
    Optimized for autoregressive generation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize cached attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # KV cache
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional caching.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            use_cache: Whether to use/update cache
            
        Returns:
            Output and optionally updated cache
        """
        batch_size, seq_len, _ = query.shape
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use cache if available
        if use_cache and self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        # Update cache
        if use_cache:
            self.kv_cache = (k, v)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, self.kv_cache if use_cache else None
    
    def reset_cache(self):
        """Reset KV cache."""
        self.kv_cache = None
