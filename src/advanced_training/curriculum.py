"""
Curriculum Learning for NOVA

Implements progressive training strategies that start with easier examples
and gradually increase difficulty, mimicking human learning.

Key Concepts:
- Progressive Difficulty: Start easy, gradually increase complexity
- Competence-Based Pacing: Advance based on model performance
- Multi-Stage Training: Distinct phases with different objectives
- Difficulty Scoring: Automatic assessment of example difficulty
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


@dataclass
class CurriculumStage:
    """Definition of a curriculum learning stage."""
    
    name: str
    difficulty_range: Tuple[float, float]  # (min, max) difficulty scores
    min_accuracy: float  # Minimum accuracy to advance
    max_epochs: int  # Maximum epochs for this stage
    learning_rate: Optional[float] = None  # Stage-specific LR
    data_fraction: float = 1.0  # Fraction of data to use
    
    def __repr__(self) -> str:
        return (
            f"CurriculumStage(name='{self.name}', "
            f"difficulty={self.difficulty_range}, "
            f"min_accuracy={self.min_accuracy:.2f})"
        )


class CurriculumStrategy(ABC):
    """
    Base class for curriculum learning strategies.
    
    A curriculum strategy determines which examples to show
    to the model at each training step, based on difficulty
    and current model competence.
    """
    
    def __init__(self, name: str = "base_curriculum"):
        """
        Initialize curriculum strategy.
        
        Args:
            name: Strategy name for logging
        """
        self.name = name
        self.current_step = 0
        self.history = []
    
    @abstractmethod
    def select_examples(
        self,
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        model_performance: float,
    ) -> List[int]:
        """
        Select examples for current training step.
        
        Args:
            examples: List of training examples
            difficulties: Difficulty score for each example
            model_performance: Current model accuracy/performance
            
        Returns:
            Indices of selected examples
        """
        pass
    
    @abstractmethod
    def should_advance(
        self,
        metrics: Dict[str, float],
        epoch: int,
    ) -> bool:
        """
        Determine if curriculum should advance to next stage.
        
        Args:
            metrics: Current training metrics
            epoch: Current epoch number
            
        Returns:
            True if should advance to next stage
        """
        pass
    
    def update(self, step: int, metrics: Dict[str, float]):
        """Update strategy state with latest metrics."""
        self.current_step = step
        self.history.append({
            'step': step,
            'metrics': metrics.copy(),
        })
    
    def reset(self):
        """Reset strategy to initial state."""
        self.current_step = 0
        self.history = []


class ProgressiveDifficulty(CurriculumStrategy):
    """
    Progressive difficulty curriculum.
    
    Gradually increases difficulty threshold over training.
    Simple but effective strategy:
    1. Start with easiest examples
    2. Linearly increase difficulty threshold
    3. Eventually include all examples
    
    Example:
        >>> curriculum = ProgressiveDifficulty(
        ...     initial_difficulty=0.2,
        ...     final_difficulty=1.0,
        ...     num_steps=10000
        ... )
        >>> indices = curriculum.select_examples(examples, difficulties, accuracy)
    """
    
    def __init__(
        self,
        initial_difficulty: float = 0.2,
        final_difficulty: float = 1.0,
        num_steps: int = 10000,
        warmup_steps: int = 1000,
    ):
        """
        Initialize progressive difficulty curriculum.
        
        Args:
            initial_difficulty: Starting difficulty threshold (0-1)
            final_difficulty: Final difficulty threshold (0-1)
            num_steps: Number of steps to reach final difficulty
            warmup_steps: Steps to stay at initial difficulty
        """
        super().__init__("progressive_difficulty")
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
    
    def get_current_threshold(self) -> float:
        """Calculate current difficulty threshold."""
        if self.current_step < self.warmup_steps:
            return self.initial_difficulty
        
        progress = min(
            (self.current_step - self.warmup_steps) / (self.num_steps - self.warmup_steps),
            1.0
        )
        
        # Linear interpolation
        threshold = (
            self.initial_difficulty +
            progress * (self.final_difficulty - self.initial_difficulty)
        )
        
        return threshold
    
    def select_examples(
        self,
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        model_performance: float,
    ) -> List[int]:
        """Select examples below current difficulty threshold."""
        threshold = self.get_current_threshold()
        
        # Select examples with difficulty <= threshold
        selected_indices = [
            i for i, diff in enumerate(difficulties)
            if diff <= threshold
        ]
        
        # Ensure at least some examples are selected
        if len(selected_indices) < len(examples) * 0.1:
            # If too few selected, take easiest 10%
            sorted_indices = sorted(
                range(len(difficulties)),
                key=lambda i: difficulties[i]
            )
            selected_indices = sorted_indices[:max(1, len(examples) // 10)]
        
        return selected_indices
    
    def should_advance(
        self,
        metrics: Dict[str, float],
        epoch: int,
    ) -> bool:
        """Check if reached final difficulty."""
        return self.get_current_threshold() >= self.final_difficulty


class CompetenceBasedPacing(CurriculumStrategy):
    """
    Competence-based curriculum pacing.
    
    Adapts difficulty based on model's actual performance:
    - Model doing well → Increase difficulty faster
    - Model struggling → Slow down progression
    - Model failing → Reduce difficulty
    
    More adaptive than fixed schedules.
    """
    
    def __init__(
        self,
        target_accuracy: float = 0.7,
        difficulty_step: float = 0.05,
        patience: int = 3,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
    ):
        """
        Initialize competence-based pacing.
        
        Args:
            target_accuracy: Target accuracy for current difficulty
            difficulty_step: How much to adjust difficulty
            patience: Epochs to wait before adjusting
            min_difficulty: Minimum difficulty threshold
            max_difficulty: Maximum difficulty threshold
        """
        super().__init__("competence_based")
        self.target_accuracy = target_accuracy
        self.difficulty_step = difficulty_step
        self.patience = patience
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        self.current_difficulty = min_difficulty
        self.epochs_at_difficulty = 0
        self.best_accuracy = 0.0
    
    def select_examples(
        self,
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        model_performance: float,
    ) -> List[int]:
        """Select examples within current difficulty window."""
        # Select examples within ±difficulty_step of current threshold
        window = self.difficulty_step * 2
        
        selected_indices = [
            i for i, diff in enumerate(difficulties)
            if abs(diff - self.current_difficulty) <= window
        ]
        
        # Fallback: take closest examples
        if len(selected_indices) < 10:
            sorted_indices = sorted(
                range(len(difficulties)),
                key=lambda i: abs(difficulties[i] - self.current_difficulty)
            )
            selected_indices = sorted_indices[:max(10, len(examples) // 20)]
        
        return selected_indices
    
    def should_advance(
        self,
        metrics: Dict[str, float],
        epoch: int,
    ) -> bool:
        """Adjust difficulty based on performance."""
        accuracy = metrics.get('accuracy', 0.0)
        self.epochs_at_difficulty += 1
        
        # Track best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        # Wait for patience epochs
        if self.epochs_at_difficulty < self.patience:
            return False
        
        # Adjust difficulty based on performance
        if accuracy > self.target_accuracy + 0.1:
            # Doing very well - increase difficulty faster
            self.current_difficulty = min(
                self.current_difficulty + self.difficulty_step * 2,
                self.max_difficulty
            )
            self.epochs_at_difficulty = 0
            self.best_accuracy = 0.0
            return True
        
        elif accuracy > self.target_accuracy:
            # Meeting target - gradually increase
            self.current_difficulty = min(
                self.current_difficulty + self.difficulty_step,
                self.max_difficulty
            )
            self.epochs_at_difficulty = 0
            self.best_accuracy = 0.0
            return True
        
        elif accuracy < self.target_accuracy - 0.2:
            # Struggling - reduce difficulty
            self.current_difficulty = max(
                self.current_difficulty - self.difficulty_step,
                self.min_difficulty
            )
            self.epochs_at_difficulty = 0
            self.best_accuracy = 0.0
            return True
        
        # Continue at current difficulty
        return False


class MultiStageCurriculum(CurriculumStrategy):
    """
    Multi-stage curriculum with distinct phases.
    
    Defines explicit training stages with clear objectives:
    - Stage 1: Basic concepts (easy examples)
    - Stage 2: Intermediate complexity
    - Stage 3: Advanced topics (hard examples)
    - Stage 4: Mixed difficulty (final stage)
    
    Each stage has:
    - Difficulty range
    - Minimum performance threshold
    - Maximum epochs
    - Optional stage-specific learning rate
    """
    
    def __init__(self, stages: List[CurriculumStage]):
        """
        Initialize multi-stage curriculum.
        
        Args:
            stages: List of curriculum stages in order
        """
        super().__init__("multi_stage")
        self.stages = stages
        self.current_stage_idx = 0
        self.epochs_in_stage = 0
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def select_examples(
        self,
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        model_performance: float,
    ) -> List[int]:
        """Select examples within current stage's difficulty range."""
        stage = self.current_stage
        min_diff, max_diff = stage.difficulty_range
        
        # Select examples within stage difficulty range
        selected_indices = [
            i for i, diff in enumerate(difficulties)
            if min_diff <= diff <= max_diff
        ]
        
        # Apply data fraction
        if stage.data_fraction < 1.0:
            num_select = int(len(selected_indices) * stage.data_fraction)
            np.random.shuffle(selected_indices)
            selected_indices = selected_indices[:num_select]
        
        return selected_indices
    
    def should_advance(
        self,
        metrics: Dict[str, float],
        epoch: int,
    ) -> bool:
        """Check if should advance to next stage."""
        stage = self.current_stage
        self.epochs_in_stage += 1
        
        accuracy = metrics.get('accuracy', 0.0)
        
        # Advance if:
        # 1. Met accuracy threshold AND
        # 2. (Completed min epochs OR reached max epochs)
        should_advance = (
            accuracy >= stage.min_accuracy or
            self.epochs_in_stage >= stage.max_epochs
        )
        
        if should_advance and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.epochs_in_stage = 0
            print(f"\n🎓 Advanced to stage: {self.current_stage.name}")
            return True
        
        return False
    
    def get_learning_rate(self) -> Optional[float]:
        """Get learning rate for current stage."""
        return self.current_stage.learning_rate
    
    @classmethod
    def create_default(cls) -> "MultiStageCurriculum":
        """Create default 4-stage curriculum."""
        stages = [
            CurriculumStage(
                name="Stage 1: Basics",
                difficulty_range=(0.0, 0.3),
                min_accuracy=0.7,
                max_epochs=5,
                learning_rate=1e-4,
            ),
            CurriculumStage(
                name="Stage 2: Intermediate",
                difficulty_range=(0.2, 0.6),
                min_accuracy=0.65,
                max_epochs=10,
                learning_rate=5e-5,
            ),
            CurriculumStage(
                name="Stage 3: Advanced",
                difficulty_range=(0.5, 0.9),
                min_accuracy=0.60,
                max_epochs=15,
                learning_rate=2e-5,
            ),
            CurriculumStage(
                name="Stage 4: Mixed",
                difficulty_range=(0.0, 1.0),
                min_accuracy=0.55,
                max_epochs=20,
                learning_rate=1e-5,
            ),
        ]
        return cls(stages)


class DifficultyScorer:
    """
    Automatic difficulty scoring for training examples.
    
    Estimates difficulty based on multiple factors:
    - Sequence length (longer = harder)
    - Vocabulary complexity (rare words = harder)
    - Syntactic complexity (nested structures = harder)
    - Model confidence (low confidence = harder)
    
    Difficulty scores range from 0 (easiest) to 1 (hardest).
    """
    
    def __init__(
        self,
        length_weight: float = 0.3,
        vocab_weight: float = 0.3,
        confidence_weight: float = 0.4,
    ):
        """
        Initialize difficulty scorer.
        
        Args:
            length_weight: Weight for sequence length factor
            vocab_weight: Weight for vocabulary complexity
            confidence_weight: Weight for model confidence
        """
        self.length_weight = length_weight
        self.vocab_weight = vocab_weight
        self.confidence_weight = confidence_weight
        
        # Statistics for normalization
        self.length_stats = {'mean': 50, 'std': 20}
        self.vocab_stats = {'mean': 0.5, 'std': 0.2}
    
    def score_length(self, sequence: List[int]) -> float:
        """Score difficulty based on sequence length."""
        length = len(sequence)
        
        # Normalize using z-score, clip to [0, 1]
        normalized = (length - self.length_stats['mean']) / self.length_stats['std']
        score = max(0.0, min(1.0, (normalized + 2) / 4))  # Map [-2, 2] to [0, 1]
        
        return score
    
    def score_vocabulary(
        self,
        sequence: List[int],
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> float:
        """Score difficulty based on vocabulary complexity."""
        if token_frequencies is None:
            return 0.5  # Default if no frequency info
        
        # Calculate average token rarity
        rarities = []
        for token in sequence:
            freq = token_frequencies.get(token, 1e-6)
            rarity = -np.log(freq + 1e-10)  # Log rarity
            rarities.append(rarity)
        
        if not rarities:
            return 0.5
        
        avg_rarity = np.mean(rarities)
        
        # Normalize
        normalized = (avg_rarity - self.vocab_stats['mean']) / self.vocab_stats['std']
        score = max(0.0, min(1.0, (normalized + 2) / 4))
        
        return score
    
    def score_confidence(
        self,
        model: nn.Module,
        sequence: torch.Tensor,
        device: str = "cpu"
    ) -> float:
        """
        Score difficulty based on model confidence.
        
        Lower confidence = harder example
        """
        model.eval()
        
        with torch.no_grad():
            input_ids = sequence.unsqueeze(0).to(device)
            outputs = model(input_ids)
            
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate average confidence (max probability per position)
            confidences = probs.max(dim=-1)[0]
            avg_confidence = confidences.mean().item()
        
        # Invert: low confidence = high difficulty
        difficulty = 1.0 - avg_confidence
        
        return difficulty
    
    def score_example(
        self,
        example: Dict[str, Any],
        model: Optional[nn.Module] = None,
        token_frequencies: Optional[Dict[int, float]] = None,
        device: str = "cpu"
    ) -> float:
        """
        Compute overall difficulty score for example.
        
        Args:
            example: Training example with 'input_ids' key
            model: Optional model for confidence scoring
            token_frequencies: Optional token frequency dict
            device: Device for computation
            
        Returns:
            Difficulty score in [0, 1]
        """
        sequence = example['input_ids']
        
        # Convert to list if tensor
        if isinstance(sequence, torch.Tensor):
            sequence_list = sequence.tolist()
        else:
            sequence_list = sequence
        
        # Length difficulty
        length_score = self.score_length(sequence_list)
        
        # Vocabulary difficulty
        vocab_score = self.score_vocabulary(sequence_list, token_frequencies)
        
        # Confidence difficulty (if model provided)
        if model is not None:
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.tensor(sequence)
            confidence_score = self.score_confidence(model, sequence, device)
        else:
            confidence_score = 0.5  # Default
        
        # Weighted combination
        total_score = (
            self.length_weight * length_score +
            self.vocab_weight * vocab_score +
            self.confidence_weight * confidence_score
        )
        
        return total_score
    
    def score_dataset(
        self,
        dataset: List[Dict[str, Any]],
        model: Optional[nn.Module] = None,
        token_frequencies: Optional[Dict[int, float]] = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> List[float]:
        """
        Score all examples in dataset.
        
        Args:
            dataset: List of training examples
            model: Optional model for confidence scoring
            token_frequencies: Optional token frequencies
            device: Device for computation
            batch_size: Batch size for model inference
            
        Returns:
            List of difficulty scores
        """
        scores = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            for example in batch:
                score = self.score_example(
                    example,
                    model=model,
                    token_frequencies=token_frequencies,
                    device=device
                )
                scores.append(score)
        
        return scores
    
    def update_statistics(self, dataset: List[Dict[str, Any]]):
        """Update normalization statistics from dataset."""
        lengths = []
        
        for example in dataset:
            sequence = example['input_ids']
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            lengths.append(len(sequence))
        
        # Update length statistics
        self.length_stats['mean'] = np.mean(lengths)
        self.length_stats['std'] = np.std(lengths)
        
        print(f"Updated difficulty scorer statistics:")
        print(f"  Length: mean={self.length_stats['mean']:.1f}, std={self.length_stats['std']:.1f}")


def create_curriculum(
    strategy: str = "progressive",
    **kwargs
) -> CurriculumStrategy:
    """
    Factory function to create curriculum strategy.
    
    Args:
        strategy: Strategy type ('progressive', 'competence', 'multi_stage')
        **kwargs: Strategy-specific arguments
        
    Returns:
        CurriculumStrategy instance
    """
    if strategy == "progressive":
        return ProgressiveDifficulty(**kwargs)
    elif strategy == "competence":
        return CompetenceBasedPacing(**kwargs)
    elif strategy == "multi_stage":
        if 'stages' in kwargs:
            return MultiStageCurriculum(kwargs['stages'])
        else:
            return MultiStageCurriculum.create_default()
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")
