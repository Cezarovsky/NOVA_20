"""
TensorBoard Integration for NOVA

Real-time metrics visualization and tracking during training.
Integrates with PyTorch's SummaryWriter for comprehensive logging.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from datetime import datetime


class TensorBoardLogger:
    """
    TensorBoard logger for NOVA training.
    
    Logs:
    - Scalar metrics (loss, perplexity, accuracy)
    - Histograms (weights, gradients)
    - Embeddings (via projector)
    - Text samples
    - Model graphs
    
    Example:
        >>> logger = TensorBoardLogger(log_dir="runs/experiment1")
        >>> logger.log_scalar("loss", loss, step=100)
        >>> logger.log_histogram("weights", model.parameters(), step=100)
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        comment: str = "",
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment (default: timestamp)
            comment: Additional info for run name
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_name = f"{experiment_name}_{comment}" if comment else experiment_name
        self.log_path = Path(log_dir) / run_name
        
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        
        print(f"TensorBoard logging to: {self.log_path}")
        print(f"Start TensorBoard: tensorboard --logdir={log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        group: Optional[str] = None,
    ):
        """
        Log scalar metric.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Training step
            group: Optional group name for organization
        """
        full_tag = f"{group}/{tag}" if group else tag
        self.writer.add_scalar(full_tag, value, step)
    
    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: int,
    ):
        """
        Log multiple related scalars.
        
        Args:
            tag: Group name
            values: Dictionary of metric names -> values
            step: Training step
        """
        self.writer.add_scalars(tag, values, step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train",
    ):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            prefix: Prefix for organization (train/val/test)
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step, group=prefix)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int,
        bins: str = "auto",
    ):
        """
        Log histogram of tensor values.
        
        Args:
            tag: Histogram name
            values: Tensor to visualize
            step: Training step
            bins: Binning strategy
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_model_weights(
        self,
        model: torch.nn.Module,
        step: int,
    ):
        """
        Log all model weights as histograms.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"weights/{name}",
                    param.data.detach().cpu().numpy(),
                    step
                )
    
    def log_gradients(
        self,
        model: torch.nn.Module,
        step: int,
    ):
        """
        Log gradient distributions.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}",
                    param.grad.detach().cpu().numpy(),
                    step
                )
    
    def log_embeddings(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[List[str]] = None,
        label_img: Optional[torch.Tensor] = None,
        global_step: int = 0,
        tag: str = "embeddings",
    ):
        """
        Log embeddings for visualization in TensorBoard projector.
        
        Args:
            embeddings: Embedding vectors [N, D]
            metadata: Labels for each embedding
            label_img: Optional images for each point
            global_step: Training step
            tag: Tag for embeddings
        """
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            label_img=label_img,
            global_step=global_step,
            tag=tag,
        )
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ):
        """
        Log text samples.
        
        Args:
            tag: Text identifier
            text: Text content
            step: Training step
        """
        self.writer.add_text(tag, text, step)
    
    def log_generation_samples(
        self,
        prompts: List[str],
        generations: List[str],
        step: int,
    ):
        """
        Log generation samples for qualitative evaluation.
        
        Args:
            prompts: Input prompts
            generations: Generated texts
            step: Training step
        """
        for idx, (prompt, gen) in enumerate(zip(prompts, generations)):
            text = f"**Prompt:** {prompt}\n\n**Generation:** {gen}"
            self.writer.add_text(f"generations/sample_{idx}", text, step)
    
    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_sample: torch.Tensor,
    ):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_sample: Sample input for tracing
        """
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_learning_rate(
        self,
        lr: float,
        step: int,
    ):
        """
        Log learning rate.
        
        Args:
            lr: Current learning rate
            step: Training step
        """
        self.log_scalar("learning_rate", lr, step, group="optimization")
    
    def log_training_summary(
        self,
        epoch: int,
        metrics: Dict[str, float],
        step: int,
    ):
        """
        Log comprehensive training summary.
        
        Args:
            epoch: Current epoch
            metrics: All metrics
            step: Training step
        """
        # Scalars
        self.log_metrics(metrics, step, prefix="train")
        
        # Summary text
        summary = f"## Epoch {epoch}\n\n"
        for name, value in metrics.items():
            summary += f"- **{name}**: {value:.4f}\n"
        
        self.log_text("training_summary", summary, step)
    
    def log_validation_summary(
        self,
        epoch: int,
        metrics: Dict[str, float],
        step: int,
    ):
        """
        Log validation summary.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            step: Training step
        """
        self.log_metrics(metrics, step, prefix="validation")
        
        # Summary text
        summary = f"## Validation - Epoch {epoch}\n\n"
        for name, value in metrics.items():
            summary += f"- **{name}**: {value:.4f}\n"
        
        self.log_text("validation_summary", summary, step)
    
    def log_attention_weights(
        self,
        attention_weights: torch.Tensor,
        step: int,
        layer: int = 0,
        head: int = 0,
    ):
        """
        Log attention weight heatmap.
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len]
            step: Training step
            layer: Layer index
            head: Attention head index
        """
        import matplotlib.pyplot as plt
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        
        weights = attention_weights.detach().cpu().numpy()
        im = ax.imshow(weights, cmap='viridis')
        
        ax.set_title(f"Attention Weights - Layer {layer}, Head {head}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        plt.colorbar(im, ax=ax)
        
        # Log to TensorBoard
        self.writer.add_figure(
            f"attention/layer_{layer}_head_{head}",
            fig,
            step
        )
        
        plt.close(fig)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"TensorBoard logging closed: {self.log_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MetricsDashboard:
    """
    Real-time metrics dashboard for training monitoring.
    
    Provides live updates and visualizations during training.
    """
    
    def __init__(
        self,
        logger: TensorBoardLogger,
        update_frequency: int = 10,
    ):
        """
        Initialize dashboard.
        
        Args:
            logger: TensorBoard logger
            update_frequency: Update every N steps
        """
        self.logger = logger
        self.update_frequency = update_frequency
        
        self.step = 0
        self.accumulated_metrics = {}
    
    def update(self, metrics: Dict[str, float], force: bool = False):
        """
        Update metrics (accumulates until update frequency reached).
        
        Args:
            metrics: Current metrics
            force: Force update regardless of frequency
        """
        # Accumulate
        for name, value in metrics.items():
            if name not in self.accumulated_metrics:
                self.accumulated_metrics[name] = []
            self.accumulated_metrics[name].append(value)
        
        self.step += 1
        
        # Update if frequency reached or forced
        if self.step % self.update_frequency == 0 or force:
            self._flush()
    
    def _flush(self):
        """Flush accumulated metrics to TensorBoard."""
        if not self.accumulated_metrics:
            return
        
        # Average accumulated metrics
        averaged = {}
        for name, values in self.accumulated_metrics.items():
            averaged[name] = np.mean(values)
        
        # Log to TensorBoard
        self.logger.log_metrics(averaged, self.step, prefix="train")
        
        # Clear accumulator
        self.accumulated_metrics.clear()
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """
        Log end-of-epoch summary.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        self.logger.log_training_summary(epoch, train_metrics, self.step)
        self.logger.log_validation_summary(epoch, val_metrics, self.step)
        
        # Compare train vs val
        if 'loss' in train_metrics and 'loss' in val_metrics:
            self.logger.log_scalars(
                "loss_comparison",
                {
                    "train": train_metrics['loss'],
                    "validation": val_metrics['loss'],
                },
                self.step
            )
