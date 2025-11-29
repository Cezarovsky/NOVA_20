"""
Advanced Data Scheduling for NOVA

Sophisticated data sampling and batch composition strategies:
- Difficulty-based sampling
- Dynamic batch composition
- Staged data loading
- Importance sampling
"""

import torch
from torch.utils.data import Sampler, DataLoader, Dataset
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
from collections import defaultdict


class DifficultyBasedSampler(Sampler):
    """
    Sample examples based on difficulty and curriculum stage.
    
    Ensures batches contain examples of appropriate difficulty
    for current training stage.
    """
    
    def __init__(
        self,
        difficulties: List[float],
        batch_size: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
    ):
        """
        Initialize difficulty-based sampler.
        
        Args:
            difficulties: Difficulty score for each example
            batch_size: Batch size
            difficulty_range: (min, max) difficulty to sample
            shuffle: Shuffle examples within difficulty range
        """
        self.difficulties = np.array(difficulties)
        self.batch_size = batch_size
        self.difficulty_range = difficulty_range
        self.shuffle = shuffle
        
        # Find valid indices
        self.valid_indices = self._get_valid_indices()
    
    def _get_valid_indices(self) -> np.ndarray:
        """Get indices within difficulty range."""
        min_diff, max_diff = self.difficulty_range
        
        valid = np.where(
            (self.difficulties >= min_diff) &
            (self.difficulties <= max_diff)
        )[0]
        
        return valid
    
    def update_difficulty_range(self, difficulty_range: tuple[float, float]):
        """Update difficulty range (for curriculum learning)."""
        self.difficulty_range = difficulty_range
        self.valid_indices = self._get_valid_indices()
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over example indices."""
        indices = self.valid_indices.copy()
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield indices
        for idx in indices:
            yield int(idx)
    
    def __len__(self) -> int:
        """Number of valid examples."""
        return len(self.valid_indices)


class DynamicBatchComposer:
    """
    Compose batches dynamically based on multiple criteria:
    - Difficulty distribution
    - Domain balance
    - Length similarity (for efficient padding)
    
    Creates more efficient batches than random sampling.
    """
    
    def __init__(
        self,
        batch_size: int,
        difficulty_mode: str = "mixed",  # "uniform", "mixed", "progressive"
        domain_balance: bool = False,
        length_grouping: bool = True,
        length_tolerance: float = 0.2,
    ):
        """
        Initialize batch composer.
        
        Args:
            batch_size: Target batch size
            difficulty_mode: How to mix difficulties in batch
            domain_balance: Balance domains within batch
            length_grouping: Group similar lengths
            length_tolerance: Max length variation (as fraction)
        """
        self.batch_size = batch_size
        self.difficulty_mode = difficulty_mode
        self.domain_balance = domain_balance
        self.length_grouping = length_grouping
        self.length_tolerance = length_tolerance
    
    def compose_batch(
        self,
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        domains: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Compose a batch from available examples.
        
        Args:
            examples: Pool of examples
            difficulties: Difficulty scores
            domains: Optional domain labels
            
        Returns:
            Indices of selected examples
        """
        if len(examples) <= self.batch_size:
            return list(range(len(examples)))
        
        # Start with a seed example
        seed_idx = np.random.randint(len(examples))
        selected = [seed_idx]
        
        # Select remaining examples
        while len(selected) < self.batch_size:
            scores = self._score_candidates(
                seed_idx,
                list(range(len(examples))),
                selected,
                examples,
                difficulties,
                domains,
            )
            
            # Pick best candidate
            best_idx = max(
                range(len(scores)),
                key=lambda i: scores[i] if i not in selected else -1
            )
            
            if best_idx in selected:
                break
            
            selected.append(best_idx)
        
        return selected
    
    def _score_candidates(
        self,
        seed_idx: int,
        candidates: List[int],
        selected: List[int],
        examples: List[Dict[str, Any]],
        difficulties: List[float],
        domains: Optional[List[int]],
    ) -> List[float]:
        """Score how well each candidate fits the batch."""
        scores = []
        
        seed_length = len(examples[seed_idx]['input_ids'])
        seed_difficulty = difficulties[seed_idx]
        
        for idx in candidates:
            if idx in selected:
                scores.append(-1.0)
                continue
            
            score = 1.0
            
            # Length similarity
            if self.length_grouping:
                example_length = len(examples[idx]['input_ids'])
                length_diff = abs(example_length - seed_length) / seed_length
                
                if length_diff < self.length_tolerance:
                    score *= 2.0
                else:
                    score *= 0.5
            
            # Difficulty matching
            if self.difficulty_mode == "uniform":
                # Prefer similar difficulty
                diff_similarity = 1.0 - abs(difficulties[idx] - seed_difficulty)
                score *= diff_similarity
            
            elif self.difficulty_mode == "mixed":
                # Want variety
                if len(selected) > 0:
                    selected_diffs = [difficulties[s] for s in selected]
                    # Prefer different difficulty
                    diversity = min(abs(difficulties[idx] - d) for d in selected_diffs)
                    score *= (1.0 + diversity)
            
            # Domain balance
            if self.domain_balance and domains is not None:
                selected_domains = [domains[s] for s in selected]
                domain_counts = defaultdict(int)
                for d in selected_domains:
                    domain_counts[d] += 1
                
                # Prefer underrepresented domains
                example_domain = domains[idx]
                domain_penalty = domain_counts[example_domain] / (len(selected) + 1)
                score *= (1.0 - domain_penalty)
            
            scores.append(score)
        
        return scores


class StagedDataLoader:
    """
    Data loader that switches between dataset stages.
    
    Useful for curriculum learning with distinct phases.
    Each stage can have different:
    - Data subset
    - Batch size
    - Sampling strategy
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        stage_names: List[str],
        batch_sizes: Optional[List[int]] = None,
        shuffle: bool = True,
    ):
        """
        Initialize staged data loader.
        
        Args:
            datasets: List of datasets (one per stage)
            stage_names: Names of each stage
            batch_sizes: Optional different batch size per stage
            shuffle: Shuffle data
        """
        self.datasets = datasets
        self.stage_names = stage_names
        self.shuffle = shuffle
        
        if batch_sizes is None:
            batch_sizes = [32] * len(datasets)
        
        self.batch_sizes = batch_sizes
        self.current_stage = 0
        
        # Create dataloaders
        self.loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            for dataset, batch_size in zip(datasets, batch_sizes)
        ]
    
    def advance_stage(self):
        """Move to next stage."""
        if self.current_stage < len(self.datasets) - 1:
            self.current_stage += 1
            print(f"\n📊 Advanced to data stage: {self.stage_names[self.current_stage]}")
    
    def get_current_loader(self) -> DataLoader:
        """Get DataLoader for current stage."""
        return self.loaders[self.current_stage]
    
    def __iter__(self):
        """Iterate over current stage's data."""
        return iter(self.get_current_loader())
    
    def __len__(self):
        """Number of batches in current stage."""
        return len(self.get_current_loader())


class ImportanceSampler(Sampler):
    """
    Sample examples based on importance weights.
    
    Useful for:
    - Focusing on hard examples
    - Balancing rare classes
    - Boosting underperforming examples
    """
    
    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
    ):
        """
        Initialize importance sampler.
        
        Args:
            weights: Importance weight for each example
            num_samples: Number of samples to draw
            replacement: Sample with replacement
        """
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.replacement = replacement
    
    def update_weights(self, new_weights: List[float]):
        """Update importance weights (e.g., based on loss)."""
        self.weights = torch.tensor(new_weights, dtype=torch.float)
    
    def __iter__(self) -> Iterator[int]:
        """Sample indices according to importance weights."""
        # Normalize weights
        probs = self.weights / self.weights.sum()
        
        # Sample indices
        indices = torch.multinomial(
            probs,
            self.num_samples,
            replacement=self.replacement
        )
        
        for idx in indices:
            yield int(idx)
    
    def __len__(self) -> int:
        """Number of samples."""
        return self.num_samples


class BalancedBatchSampler(Sampler):
    """
    Create balanced batches across classes/domains.
    
    Ensures each batch has roughly equal representation
    of different categories.
    """
    
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = False,
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Class/domain label for each example
            batch_size: Batch size
            drop_last: Drop incomplete batches
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = batch_size // self.num_labels
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield balanced batches."""
        # Shuffle indices within each label
        label_iters = {}
        for label, indices in self.label_to_indices.items():
            indices = indices.copy()
            np.random.shuffle(indices)
            label_iters[label] = iter(indices)
        
        # Create batches
        while True:
            batch = []
            
            # Sample from each label
            for label in sorted(self.label_to_indices.keys()):
                label_iter = label_iters[label]
                
                for _ in range(self.samples_per_label):
                    try:
                        idx = next(label_iter)
                        batch.append(idx)
                    except StopIteration:
                        # Reshuffle when exhausted
                        indices = self.label_to_indices[label].copy()
                        np.random.shuffle(indices)
                        label_iters[label] = iter(indices)
                        
                        if not self.drop_last:
                            idx = next(label_iters[label])
                            batch.append(idx)
            
            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch
            
            if len(batch) < self.batch_size:
                break
    
    def __len__(self) -> int:
        """Number of batches."""
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def create_curriculum_dataloader(
    dataset: Dataset,
    difficulties: List[float],
    difficulty_range: tuple[float, float],
    batch_size: int,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with curriculum-aware sampling.
    
    Args:
        dataset: Training dataset
        difficulties: Difficulty score per example
        difficulty_range: (min, max) difficulty to include
        batch_size: Batch size
        shuffle: Shuffle data
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with difficulty-based sampling
    """
    sampler = DifficultyBasedSampler(
        difficulties=difficulties,
        batch_size=batch_size,
        difficulty_range=difficulty_range,
        shuffle=shuffle,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **kwargs
    )
    
    return loader
