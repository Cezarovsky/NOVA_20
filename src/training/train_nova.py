"""
NOVA Training Loop

Main training script with next-token prediction objective.
Uses AI2AI embeddings for fast knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import time
from tqdm import tqdm
import json

from ..ml.transformer import TransformerModel
from .dataset import NovaDataset, NovaDataLoader
from .corpus_processor import CorpusProcessor


class NovaTrainer:
    """
    NOVA model trainer.
    
    Handles:
    - Next-token prediction training
    - AI2AI embedding-based training
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Validation
    """
    
    def __init__(
        self,
        model: TransformerModel,
        train_loader: NovaDataLoader,
        val_loader: Optional[NovaDataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: NOVA transformer model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Learning rate warmup steps
            max_grad_norm: Max gradient norm for clipping
            gradient_accumulation_steps: Accumulate gradients over multiple steps
            mixed_precision: Use mixed precision (FP16) training
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.1, (warmup_steps / step) ** 0.5)
        
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Statistics
        self.train_history = {
            "loss": [],
            "perplexity": [],
            "learning_rate": [],
        }
        self.val_history = {
            "loss": [],
            "perplexity": [],
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            
            epoch_loss += loss
            epoch_steps += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'ppl': f'{torch.exp(torch.tensor(loss)):.2f}',
                'lr': f'{current_lr:.2e}',
            })
        
        # Compute epoch metrics
        avg_loss = epoch_loss / epoch_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
        
        # Update history
        self.train_history["loss"].append(avg_loss)
        self.train_history["perplexity"].append(perplexity)
        self.train_history["learning_rate"].append(metrics["learning_rate"])
        
        return metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Extract inputs based on data type
        if "input_ids" in batch:
            # Text mode
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask
                    )
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1)
                    )
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask
                )
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
        
        elif "embeddings" in batch:
            # Embedding mode (AI2AI)
            embeddings = batch["embeddings"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            
            # Forward pass through transformer (skip embedding layer)
            if self.mixed_precision:
                with autocast():
                    # Pass embeddings directly to transformer blocks
                    outputs = self.model.forward_embeddings(
                        embeddings,
                        attention_mask=attention_mask
                    )
                    # Compute MSE loss for embedding prediction
                    loss = nn.functional.mse_loss(
                        outputs,
                        labels,
                        reduction='mean'
                    )
            else:
                outputs = self.model.forward_embeddings(
                    embeddings,
                    attention_mask=attention_mask
                )
                loss = nn.functional.mse_loss(
                    outputs,
                    labels,
                    reduction='mean'
                )
        
        else:
            raise ValueError("Batch must contain 'input_ids' or 'embeddings'")
        
        # Normalize loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if accumulated enough gradients
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.mixed_precision:
                # Unscale and clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if "input_ids" in batch:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask")
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
            
            elif "embeddings" in batch:
                embeddings = batch["embeddings"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask")
                
                outputs = self.model.forward_embeddings(
                    embeddings,
                    attention_mask=attention_mask
                )
                loss = nn.functional.mse_loss(outputs, labels)
            
            val_loss += loss.item()
            val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        metrics = {
            "loss": avg_val_loss,
            "perplexity": val_perplexity,
        }
        
        # Update history
        self.val_history["loss"].append(avg_val_loss)
        self.val_history["perplexity"].append(val_perplexity)
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        validate_every: int = 1,
        save_every: int = 1,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting NOVA Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                  f"Perplexity: {train_metrics['perplexity']:.2f}")
            
            # Validate
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                      f"Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Early stopping
                if early_stopping_patience:
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        patience_counter = 0
                        # Save best model
                        if self.checkpoint_dir:
                            self.save_checkpoint(f"best_model.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping after {epoch + 1} epochs")
                            break
            
            # Save checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {training_time / 60:.1f} minutes")
        print(f"Final train loss: {self.train_history['loss'][-1]:.4f}")
        if self.val_history['loss']:
            print(f"Final val loss: {self.val_history['loss'][-1]:.4f}")
        print(f"{'='*60}\n")
        
        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "training_time": training_time,
        }
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")
