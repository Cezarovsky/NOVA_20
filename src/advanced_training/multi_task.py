"""
Multi-Task Learning for NOVA

Train on multiple tasks simultaneously with shared representations:
- Task-specific heads
- Shared encoders
- Dynamic task weighting
- Task scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Task:
    """Task configuration."""
    name: str
    num_classes: Optional[int] = None
    loss_fn: Optional[Callable] = None
    weight: float = 1.0
    metric_fn: Optional[Callable] = None


class TaskHead(nn.Module):
    """
    Task-specific output head.
    
    Projects shared representations to task-specific outputs.
    Each task gets its own head for specialized predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize task head.
        
        Args:
            input_dim: Input dimension (from shared encoder)
            output_dim: Output dimension (task-specific)
            hidden_dim: Optional hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Task-specific logits [batch_size, output_dim]
        """
        return self.projection(x)


class SharedEncoder(nn.Module):
    """
    Shared encoder for multi-task learning.
    
    Extracts common representations across all tasks.
    Task-specific heads then specialize these representations.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        representation_dim: int,
    ):
        """
        Initialize shared encoder.
        
        Args:
            base_model: Base model (e.g., NOVA transformer)
            representation_dim: Dimension of shared representations
        """
        super().__init__()
        
        self.base_model = base_model
        self.representation_dim = representation_dim
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract shared representations.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments for base model
            
        Returns:
            Shared representations
        """
        return self.base_model(x, **kwargs)


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model.
    
    Architecture:
    - Shared encoder (common representations)
    - Task-specific heads (specialized outputs)
    
    Enables knowledge transfer between related tasks.
    """
    
    def __init__(
        self,
        shared_encoder: nn.Module,
        task_heads: Dict[str, TaskHead],
    ):
        """
        Initialize multi-task model.
        
        Args:
            shared_encoder: Shared feature extractor
            task_heads: Dictionary of task-specific heads
        """
        super().__init__()
        
        self.shared_encoder = shared_encoder
        self.task_heads = nn.ModuleDict(task_heads)
    
    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            task: Specific task to run (if None, run all tasks)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of task predictions
        """
        # Extract shared representations
        shared_repr = self.shared_encoder(x, **kwargs)
        
        # Run task-specific heads
        outputs = {}
        
        if task is not None:
            # Single task
            outputs[task] = self.task_heads[task](shared_repr)
        else:
            # All tasks
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(shared_repr)
        
        return outputs
    
    def freeze_encoder(self):
        """Freeze shared encoder parameters."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze shared encoder parameters."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True


class TaskWeighting:
    """
    Dynamic task weighting strategies.
    
    Automatically balances task losses during training:
    - Uncertainty weighting (multi-task learning using uncertainty)
    - GradNorm (gradient normalization)
    - Equal weighting (baseline)
    """
    
    def __init__(
        self,
        tasks: List[str],
        method: str = "uncertainty",
        alpha: float = 0.5,
    ):
        """
        Initialize task weighting.
        
        Args:
            tasks: List of task names
            method: Weighting method ("uncertainty", "gradnorm", "equal")
            alpha: GradNorm hyperparameter (restoring force)
        """
        self.tasks = tasks
        self.method = method
        self.alpha = alpha
        
        # Initialize weights
        if method == "uncertainty":
            # Learnable log-variance parameters
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in tasks
            })
        else:
            # Fixed equal weights
            self.weights = {task: 1.0 for task in tasks}
        
        # Track initial losses (for GradNorm)
        self.initial_losses: Dict[str, float] = {}
        self.loss_history: Dict[str, List[float]] = {task: [] for task in tasks}
    
    def get_weights(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate task weights.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of task weights
        """
        if self.method == "uncertainty":
            return self._uncertainty_weighting(losses)
        
        elif self.method == "gradnorm":
            return self._gradnorm_weighting(losses)
        
        else:  # equal
            return self.weights
    
    def _uncertainty_weighting(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Uncertainty-based weighting.
        
        Weight = 1 / (2 * sigma^2), where sigma is learned.
        Regularization term: log(sigma^2)
        
        Reference: Multi-Task Learning Using Uncertainty (Kendall et al., 2018)
        """
        weights = {}
        
        for task, loss in losses.items():
            log_var = self.log_vars[task]
            
            # Weight inversely proportional to uncertainty
            precision = torch.exp(-log_var)
            weights[task] = 0.5 * precision
            
            # Add regularization term to loss
            losses[task] = losses[task] + 0.5 * log_var
        
        return weights
    
    def _gradnorm_weighting(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        GradNorm-based weighting.
        
        Balances gradient magnitudes across tasks.
        Tasks with larger gradients get lower weights.
        
        Reference: GradNorm (Chen et al., 2018)
        """
        # Store initial losses
        if not self.initial_losses:
            self.initial_losses = {
                task: loss.item()
                for task, loss in losses.items()
            }
        
        # Calculate relative inverse training rates
        loss_ratios = {}
        
        for task, loss in losses.items():
            self.loss_history[task].append(loss.item())
            
            if len(self.loss_history[task]) > 1:
                # Relative decrease
                initial = self.initial_losses[task]
                current = loss.item()
                loss_ratios[task] = current / initial
            else:
                loss_ratios[task] = 1.0
        
        # Calculate average training rate
        avg_ratio = np.mean(list(loss_ratios.values()))
        
        # Target: r_i(t) = avg_ratio^alpha
        weights = {}
        
        for task in self.tasks:
            ratio = loss_ratios[task]
            target = avg_ratio ** self.alpha
            
            # Weight inversely proportional to training rate
            weights[task] = target / ratio
        
        return weights
    
    def compute_total_loss(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute weighted total loss.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Weighted sum of losses
        """
        weights = self.get_weights(losses)
        
        total_loss = sum(
            weights[task] * loss
            for task, loss in losses.items()
        )
        
        return total_loss


class TaskScheduler:
    """
    Task scheduling strategies.
    
    Determines which tasks to train on at each step:
    - Round-robin (equal sampling)
    - Proportional (sample by dataset size)
    - Performance-based (focus on worst tasks)
    """
    
    def __init__(
        self,
        tasks: List[str],
        method: str = "round_robin",
        task_sizes: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize task scheduler.
        
        Args:
            tasks: List of task names
            method: Scheduling method
            task_sizes: Dataset size per task (for proportional)
        """
        self.tasks = tasks
        self.method = method
        self.task_sizes = task_sizes or {task: 1 for task in tasks}
        
        self.current_idx = 0
        self.performance: Dict[str, float] = {task: 0.0 for task in tasks}
    
    def select_task(self) -> str:
        """
        Select next task to train on.
        
        Returns:
            Task name
        """
        if self.method == "round_robin":
            task = self.tasks[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.tasks)
            return task
        
        elif self.method == "proportional":
            # Sample proportional to dataset size
            total_size = sum(self.task_sizes.values())
            probs = [
                self.task_sizes[task] / total_size
                for task in self.tasks
            ]
            return np.random.choice(self.tasks, p=probs)
        
        elif self.method == "performance":
            # Focus on worst-performing task
            worst_task = min(self.tasks, key=lambda t: self.performance[t])
            return worst_task
        
        else:
            raise ValueError(f"Unknown scheduling method: {self.method}")
    
    def update_performance(self, task: str, metric: float):
        """
        Update task performance metric.
        
        Args:
            task: Task name
            metric: Performance metric (higher is better)
        """
        self.performance[task] = metric


class MultiTaskTrainer:
    """
    Multi-task learning trainer.
    
    Orchestrates training across multiple tasks:
    - Task scheduling
    - Loss weighting
    - Shared encoder updates
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        tasks: Dict[str, Task],
        task_weighting: TaskWeighting,
        task_scheduler: TaskScheduler,
    ):
        """
        Initialize multi-task trainer.
        
        Args:
            model: Multi-task model
            tasks: Task configurations
            task_weighting: Task weighting strategy
            task_scheduler: Task scheduling strategy
        """
        self.model = model
        self.tasks = tasks
        self.task_weighting = task_weighting
        self.task_scheduler = task_scheduler
        
        self.step_count = 0
    
    def train_step(
        self,
        batches: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batches: Dictionary of task batches
            optimizer: Optimizer
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Select task(s)
        if self.task_scheduler.method == "round_robin":
            # Train one task at a time
            task_name = self.task_scheduler.select_task()
            batch = batches[task_name]
            
            # Forward pass
            outputs = self.model(batch['input'], task=task_name)
            
            # Compute loss
            task = self.tasks[task_name]
            loss = task.loss_fn(outputs[task_name], batch['target'])
            
            losses = {task_name: loss}
        
        else:
            # Train all tasks together
            losses = {}
            
            for task_name, batch in batches.items():
                # Forward pass
                outputs = self.model(batch['input'], task=task_name)
                
                # Compute loss
                task = self.tasks[task_name]
                losses[task_name] = task.loss_fn(
                    outputs[task_name],
                    batch['target']
                )
        
        # Weighted total loss
        total_loss = self.task_weighting.compute_total_loss(losses)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        self.step_count += 1
        
        return {task: loss.item() for task, loss in losses.items()}
    
    def evaluate(
        self,
        batches: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evaluate on all tasks.
        
        Args:
            batches: Dictionary of task batches
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            for task_name, batch in batches.items():
                # Forward pass
                outputs = self.model(batch['input'], task=task_name)
                
                # Compute metric
                task = self.tasks[task_name]
                if task.metric_fn:
                    metric = task.metric_fn(
                        outputs[task_name],
                        batch['target']
                    )
                    metrics[task_name] = metric
                    
                    # Update scheduler
                    self.task_scheduler.update_performance(task_name, metric)
        
        return metrics


def create_multi_task_model(
    base_model: nn.Module,
    task_configs: List[Task],
    shared_dim: int,
    hidden_dim: Optional[int] = None,
) -> MultiTaskModel:
    """
    Factory function for multi-task model.
    
    Args:
        base_model: Base encoder model
        task_configs: List of task configurations
        shared_dim: Shared representation dimension
        hidden_dim: Optional hidden layer dimension
        
    Returns:
        Multi-task model
    """
    # Create shared encoder
    shared_encoder = SharedEncoder(base_model, shared_dim)
    
    # Create task heads
    task_heads = {}
    
    for task in task_configs:
        if task.num_classes is None:
            raise ValueError(f"Task {task.name} missing num_classes")
        
        task_heads[task.name] = TaskHead(
            input_dim=shared_dim,
            output_dim=task.num_classes,
            hidden_dim=hidden_dim,
        )
    
    return MultiTaskModel(shared_encoder, task_heads)
