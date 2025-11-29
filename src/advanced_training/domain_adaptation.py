"""
Domain Adaptation for NOVA

Transfer learning and domain-specific fine-tuning infrastructure.
Enables NOVA to specialize for specific domains (physics, math, code)
while retaining general knowledge.

Key Techniques:
- Fine-tuning: Adapt pre-trained model to new domain
- Layer freezing: Selective parameter updates
- Domain-specific layers: Specialized components per domain
- Adaptive normalization: Domain-conditioned batch/layer norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import json


class DomainAdapter:
    """
    Manages domain adaptation for NOVA models.
    
    Provides utilities for:
    - Loading pre-trained weights
    - Freezing/unfreezing layers
    - Domain-specific parameter groups
    - Gradual unfreezing schedules
    
    Example:
        >>> adapter = DomainAdapter(model)
        >>> adapter.freeze_encoder()
        >>> adapter.add_domain_head("physics")
        >>> adapter.train_domain("physics", train_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        freeze_embeddings: bool = True,
        freeze_encoder: bool = False,
        freeze_layers: Optional[List[str]] = None,
    ):
        """
        Initialize domain adapter.
        
        Args:
            model: NOVA model to adapt
            freeze_embeddings: Freeze embedding layers
            freeze_encoder: Freeze all encoder layers
            freeze_layers: Specific layer names to freeze
        """
        self.model = model
        self.frozen_layers: Set[str] = set()
        
        # Apply initial freezing
        if freeze_embeddings:
            self.freeze_embeddings()
        
        if freeze_encoder:
            self.freeze_encoder()
        
        if freeze_layers:
            for layer_name in freeze_layers:
                self.freeze_layer(layer_name)
    
    def freeze_embeddings(self):
        """Freeze embedding layers."""
        for name, param in self.model.named_parameters():
            if 'embedding' in name.lower():
                param.requires_grad = False
                self.frozen_layers.add(name)
        
        print(f"Froze {len([n for n in self.frozen_layers if 'embedding' in n.lower()])} embedding parameters")
    
    def freeze_encoder(self):
        """Freeze all encoder layers."""
        for name, param in self.model.named_parameters():
            if 'encoder' in name.lower():
                param.requires_grad = False
                self.frozen_layers.add(name)
        
        print(f"Froze encoder layers")
    
    def freeze_layer(self, layer_name: str):
        """Freeze specific layer by name."""
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = False
                self.frozen_layers.add(name)
                frozen_count += 1
        
        print(f"Froze {frozen_count} parameters matching '{layer_name}'")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.frozen_layers.clear()
        print("Unfroze all parameters")
    
    def unfreeze_layer(self, layer_name: str):
        """Unfreeze specific layer."""
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            if layer_name in name and name in self.frozen_layers:
                param.requires_grad = True
                self.frozen_layers.remove(name)
                unfrozen_count += 1
        
        print(f"Unfroze {unfrozen_count} parameters matching '{layer_name}'")
    
    def gradual_unfreeze(
        self,
        num_layers: int,
        reverse: bool = True
    ):
        """
        Gradually unfreeze layers (useful for fine-tuning).
        
        Args:
            num_layers: Number of layers to unfreeze
            reverse: Unfreeze from last layer (True) or first (False)
        """
        # Get all layer names
        layer_names = []
        for name, _ in self.model.named_parameters():
            if 'layer' in name or 'block' in name:
                # Extract layer number
                parts = name.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_names.append((int(part), name))
                        break
        
        # Sort layers
        layer_names.sort(reverse=reverse)
        
        # Unfreeze top num_layers
        unfrozen = set()
        for layer_idx, name in layer_names[:num_layers]:
            if layer_idx not in unfrozen:
                self.unfreeze_layer(f".{layer_idx}.")
                unfrozen.add(layer_idx)
    
    def get_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_parameter_groups(
        self,
        base_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups for differential learning rates.
        
        Args:
            base_lr: Learning rate for frozen/base layers
            head_lr: Learning rate for task-specific heads
            
        Returns:
            List of parameter groups for optimizer
        """
        # Separate parameters into groups
        base_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'head' in name.lower() or 'output' in name.lower():
                head_params.append(param)
            else:
                base_params.append(param)
        
        param_groups = []
        
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': base_lr,
                'name': 'base',
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'head',
            })
        
        return param_groups
    
    def save_adapter_state(self, filepath: Path):
        """Save adapter configuration."""
        state = {
            'frozen_layers': list(self.frozen_layers),
            'trainable_params': self.get_trainable_parameters(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved adapter state to {filepath}")
    
    def load_adapter_state(self, filepath: Path):
        """Load adapter configuration."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Reapply freezing
        self.unfreeze_all()
        for layer_name in state['frozen_layers']:
            for name, param in self.model.named_parameters():
                if name == layer_name:
                    param.requires_grad = False
                    self.frozen_layers.add(name)
        
        print(f"Loaded adapter state from {filepath}")


class FineTuner:
    """
    Fine-tuning wrapper for domain adaptation.
    
    Implements common fine-tuning strategies:
    - Full fine-tuning: Train all parameters
    - Partial fine-tuning: Train only certain layers
    - Progressive fine-tuning: Gradually unfreeze layers
    - Discriminative fine-tuning: Different LR per layer
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapter: DomainAdapter,
        strategy: str = "progressive",
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: Model to fine-tune
            adapter: Domain adapter instance
            strategy: Fine-tuning strategy
        """
        self.model = model
        self.adapter = adapter
        self.strategy = strategy
        self.current_epoch = 0
    
    def setup_progressive(
        self,
        total_epochs: int,
        unfreeze_schedule: Optional[List[int]] = None,
    ):
        """
        Setup progressive fine-tuning.
        
        Args:
            total_epochs: Total fine-tuning epochs
            unfreeze_schedule: Epochs at which to unfreeze layers
        """
        if unfreeze_schedule is None:
            # Default: unfreeze 1 layer every 3 epochs
            num_layers = self._count_layers()
            unfreeze_schedule = [
                i * 3 for i in range(1, num_layers + 1)
                if i * 3 < total_epochs
            ]
        
        self.unfreeze_schedule = unfreeze_schedule
        print(f"Progressive fine-tuning schedule: {unfreeze_schedule}")
    
    def _count_layers(self) -> int:
        """Count number of layers in model."""
        layer_indices = set()
        
        for name, _ in self.model.named_parameters():
            parts = name.split('.')
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
        
        return len(layer_indices)
    
    def on_epoch_start(self, epoch: int):
        """Called at start of each epoch."""
        self.current_epoch = epoch
        
        if self.strategy == "progressive":
            # Check if should unfreeze layers
            if hasattr(self, 'unfreeze_schedule'):
                if epoch in self.unfreeze_schedule:
                    # Unfreeze one more layer
                    layers_to_unfreeze = self.unfreeze_schedule.index(epoch) + 1
                    self.adapter.gradual_unfreeze(layers_to_unfreeze)
    
    def get_optimizer_params(
        self,
        base_lr: float = 1e-5,
        head_lr: float = 1e-4,
        discriminative_factor: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Get optimizer parameter groups with discriminative learning rates.
        
        Args:
            base_lr: Base learning rate (for deepest layers)
            head_lr: Learning rate for output head
            discriminative_factor: LR multiplier per layer group
            
        Returns:
            Parameter groups for optimizer
        """
        if self.strategy == "discriminative":
            # Group parameters by layer depth
            layer_groups = {}
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Determine layer group
                if 'embedding' in name.lower():
                    group = 0
                elif 'encoder' in name.lower() or 'layer' in name.lower():
                    # Extract layer number
                    for part in name.split('.'):
                        if part.isdigit():
                            group = int(part) + 1
                            break
                    else:
                        group = 1
                else:
                    group = 999  # Output/head
                
                if group not in layer_groups:
                    layer_groups[group] = []
                
                layer_groups[group].append(param)
            
            # Create parameter groups with increasing LR
            param_groups = []
            sorted_groups = sorted(layer_groups.keys())
            
            for i, group_idx in enumerate(sorted_groups):
                if group_idx == 999:
                    lr = head_lr
                else:
                    # Discriminative LR: deeper layers get lower LR
                    lr = base_lr * (discriminative_factor ** i)
                
                param_groups.append({
                    'params': layer_groups[group_idx],
                    'lr': lr,
                    'name': f'group_{group_idx}',
                })
            
            return param_groups
        
        else:
            # Standard parameter groups
            return self.adapter.get_parameter_groups(base_lr, head_lr)


class DomainDiscriminator(nn.Module):
    """
    Domain adversarial discriminator.
    
    Used for domain-adversarial training:
    - Model learns domain-invariant features
    - Discriminator tries to identify domain
    - Helps with domain transfer
    
    Based on "Domain-Adversarial Training of Neural Networks" (Ganin et al.)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_domains: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize domain discriminator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_domains: Number of domains
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict domain from features.
        
        Args:
            features: Hidden features [batch, seq_len, dim] or [batch, dim]
            
        Returns:
            Domain logits [batch, num_domains]
        """
        # Pool sequence if needed
        if features.dim() == 3:
            features = features.mean(dim=1)  # Average pooling
        
        return self.layers(features)


class AdaptiveLayerNorm(nn.Module):
    """
    Domain-adaptive layer normalization.
    
    Learns domain-specific scale and shift parameters:
    - Shared normalization statistics
    - Domain-specific affine transformation
    - Helps model adapt to different domains
    
    Similar to conditional batch normalization.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        num_domains: int = 2,
        eps: float = 1e-5,
    ):
        """
        Initialize adaptive layer norm.
        
        Args:
            normalized_shape: Feature dimension
            num_domains: Number of domains
            eps: Epsilon for numerical stability
        """
        super().__init__()
        
        self.normalized_shape = normalized_shape
        self.num_domains = num_domains
        self.eps = eps
        
        # Domain-specific parameters
        self.domain_scales = nn.Parameter(torch.ones(num_domains, normalized_shape))
        self.domain_shifts = nn.Parameter(torch.zeros(num_domains, normalized_shape))
    
    def forward(
        self,
        x: torch.Tensor,
        domain_id: int = 0,
    ) -> torch.Tensor:
        """
        Apply domain-adaptive normalization.
        
        Args:
            x: Input tensor [..., normalized_shape]
            domain_id: Domain identifier
            
        Returns:
            Normalized tensor
        """
        # Standard layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply domain-specific affine transformation
        scale = self.domain_scales[domain_id]
        shift = self.domain_shifts[domain_id]
        
        output = x_normalized * scale + shift
        
        return output


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool = False,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Load pre-trained weights with flexible matching.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Require exact match of keys
        prefix: Prefix to add to checkpoint keys
        
    Returns:
        Dictionary with loading information
    """
    print(f"\nLoading pre-trained weights from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Add prefix if needed
    if prefix:
        state_dict = {
            f"{prefix}.{k}": v
            for k, v in state_dict.items()
        }
    
    # Load with flexible matching
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=strict
    )
    
    # Report results
    info = {
        'loaded_keys': len(state_dict),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
    }
    
    print(f"  Loaded {info['loaded_keys']} keys")
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    return info
