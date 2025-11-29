"""
Advanced Optimization Strategies for NOVA

Sophisticated learning rate schedules and optimizer configurations:
- Warmup schedules
- Cosine annealing
- Adaptive learning rates
- Domain-specific optimization
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Any, Optional
import math
import numpy as np


class WarmupScheduler(_LRScheduler):
    """
    Learning rate warmup scheduler.
    
    Gradually increases LR from 0 to base_lr over warmup period.
    Prevents instability at training start with large learning rates.
    
    Example:
        >>> optimizer = Adam(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupScheduler(optimizer, warmup_steps=1000)
        >>> for step in range(num_steps):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # After warmup, return base LR
            return self.base_lrs


class CosineAnnealingScheduler(_LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    Gradually reduces LR following cosine curve.
    Helps model converge to better minima.
    
    LR(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of steps
            eta_min: Minimum learning rate
            warmup_steps: Optional warmup period
            last_epoch: Last epoch index
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        else:
            # Cosine annealing
            t = self.last_epoch - self.warmup_steps
            T = self.T_max - self.warmup_steps
            
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * t / T)) / 2
                for base_lr in self.base_lrs
            ]


class WarmupCosineScheduler(_LRScheduler):
    """
    Combined warmup + cosine annealing schedule.
    
    Best of both worlds:
    - Stable start with warmup
    - Smooth convergence with cosine annealing
    
    Most commonly used schedule for transformer training.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize warmup + cosine scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Warmup period
            total_steps: Total training steps
            min_lr: Minimum LR after annealing
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) *
                0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.
    
    Smoothly decays LR from base to end value:
    LR(t) = (base_lr - end_lr) * (1 - t/T)^power + end_lr
    
    power=1.0 gives linear decay
    power>1.0 decays faster initially
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        end_lr: float = 0.0,
        power: float = 1.0,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            total_steps: Total training steps
            end_lr: Final learning rate
            power: Polynomial power
            warmup_steps: Optional warmup
            last_epoch: Last epoch index
        """
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        else:
            # Polynomial decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                (base_lr - self.end_lr) * (1 - progress) ** self.power + self.end_lr
                for base_lr in self.base_lrs
            ]


class AdaptiveOptimizer:
    """
    Adaptive optimizer with dynamic learning rate adjustment.
    
    Monitors training metrics and adjusts LR automatically:
    - Reduce LR on plateau
    - Increase LR if loss diverges
    - Domain-specific adaptation
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",  # "min" or "max"
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            optimizer: Base optimizer
            mode: "min" for loss, "max" for accuracy
            factor: LR reduction factor
            patience: Epochs to wait before reducing
            threshold: Minimum change to qualify as improvement
            min_lr: Minimum allowed learning rate
            verbose: Print LR changes
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.epochs_since_improvement = 0
        self.history = []
    
    def step(self, metric: float):
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value (loss or accuracy)
        """
        self.history.append(metric)
        
        # Check for improvement
        is_better = (
            (self.mode == 'min' and metric < self.best_metric - self.threshold) or
            (self.mode == 'max' and metric > self.best_metric + self.threshold)
        )
        
        if is_better:
            self.best_metric = metric
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Reduce LR if no improvement
        if self.epochs_since_improvement >= self.patience:
            self._reduce_lr()
            self.epochs_since_improvement = 0
    
    def _reduce_lr(self):
        """Reduce learning rate by factor."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                
                if self.verbose:
                    print(f"\n📉 Reducing LR: {old_lr:.2e} → {new_lr:.2e}")
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class DomainSpecificOptimizer:
    """
    Optimizer with domain-specific learning rates.
    
    Different domains can have different learning rates,
    allowing fine-grained control over multi-domain training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        domain_lrs: Dict[str, float],
        default_lr: float = 1e-4,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """
        Initialize domain-specific optimizer.
        
        Args:
            model: Model to optimize
            domain_lrs: Learning rate per domain
            default_lr: Default LR for unspecified domains
            optimizer_class: Optimizer class (AdamW, Adam, SGD)
            **optimizer_kwargs: Additional optimizer arguments
        """
        self.model = model
        self.domain_lrs = domain_lrs
        self.default_lr = default_lr
        
        # Create parameter groups per domain
        param_groups = self._create_param_groups()
        
        # Initialize optimizer
        self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)
        
        self.current_domain = None
    
    def _create_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for each domain."""
        param_groups = []
        
        # Group parameters by domain affinity
        domain_params = {domain: [] for domain in self.domain_lrs}
        domain_params['default'] = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine domain from parameter name
            assigned = False
            for domain in self.domain_lrs:
                if domain.lower() in name.lower():
                    domain_params[domain].append(param)
                    assigned = True
                    break
            
            if not assigned:
                domain_params['default'].append(param)
        
        # Create parameter groups
        for domain, params in domain_params.items():
            if not params:
                continue
            
            lr = self.domain_lrs.get(domain, self.default_lr)
            
            param_groups.append({
                'params': params,
                'lr': lr,
                'name': domain,
            })
        
        return param_groups
    
    def set_domain(self, domain: str):
        """Set current domain for training."""
        self.current_domain = domain
    
    def adjust_domain_lr(self, domain: str, new_lr: float):
        """Adjust learning rate for specific domain."""
        for param_group in self.optimizer.param_groups:
            if param_group.get('name') == domain:
                param_group['lr'] = new_lr
                print(f"Adjusted {domain} LR to {new_lr:.2e}")
    
    def step(self):
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class LayerWiseLRScheduler(_LRScheduler):
    """
    Layer-wise learning rate decay (LLRD).
    
    Deeper layers get lower learning rates.
    Helps preserve pre-trained knowledge while adapting.
    
    Common in BERT fine-tuning.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_layers: int,
        decay_rate: float = 0.9,
        last_epoch: int = -1,
    ):
        """
        Initialize layer-wise LR scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            num_layers: Number of layers
            decay_rate: LR decay per layer
            last_epoch: Last epoch index
        """
        self.num_layers = num_layers
        self.decay_rate = decay_rate
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate layer-wise learning rates."""
        lrs = []
        
        for param_group in self.optimizer.param_groups:
            # Determine layer depth from param group name
            layer_idx = param_group.get('layer_idx', self.num_layers)
            
            # Decay LR based on layer depth
            decay_factor = self.decay_rate ** (self.num_layers - layer_idx)
            lr = param_group['initial_lr'] * decay_factor
            
            lrs.append(lr)
        
        return lrs


def create_optimizer_with_schedule(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    schedule: str = "warmup_cosine",
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    **kwargs
) -> tuple[Optimizer, _LRScheduler]:
    """
    Factory function for optimizer + scheduler.
    
    Args:
        model: Model to optimize
        optimizer_name: "adamw", "adam", "sgd"
        lr: Learning rate
        weight_decay: Weight decay
        schedule: Schedule type
        warmup_steps: Warmup steps
        total_steps: Total training steps
        **kwargs: Additional optimizer/scheduler arguments
        
    Returns:
        (optimizer, scheduler) tuple
    """
    # Create optimizer
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create scheduler
    if schedule == "warmup":
        scheduler = WarmupScheduler(optimizer, warmup_steps)
    
    elif schedule == "cosine":
        scheduler = CosineAnnealingScheduler(
            optimizer,
            T_max=total_steps,
            warmup_steps=warmup_steps
        )
    
    elif schedule == "warmup_cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    
    elif schedule == "polynomial":
        scheduler = PolynomialDecayScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
    
    elif schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return optimizer, scheduler
