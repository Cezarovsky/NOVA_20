"""
Comprehensive Tests for Advanced Training

Tests for:
- Curriculum learning strategies
- Domain adaptation
- Data scheduling
- Optimization schedules
- Multi-task learning
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from src.advanced_training.curriculum import (
    ProgressiveDifficulty,
    CompetenceBasedPacing,
    MultiStageCurriculum,
    CurriculumStage,
    DifficultyScorer,
)
from src.advanced_training.domain_adaptation import (
    DomainAdapter,
    FineTuner,
    DomainDiscriminator,
    AdaptiveLayerNorm,
    load_pretrained_weights,
)
from src.advanced_training.data_scheduler import (
    DifficultyBasedSampler,
    DynamicBatchComposer,
    StagedDataLoader,
    ImportanceSampler,
    BalancedBatchSampler,
)
from src.advanced_training.optimization import (
    WarmupScheduler,
    CosineAnnealingScheduler,
    WarmupCosineScheduler,
    AdaptiveOptimizer,
    create_optimizer_with_schedule,
)
from src.advanced_training.multi_task import (
    TaskHead,
    SharedEncoder,
    MultiTaskModel,
    TaskWeighting,
    TaskScheduler,
    Task,
    create_multi_task_model,
)


# Test Models
class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(100, dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.head = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x.mean(dim=1))
        return self.head(x)


# Curriculum Learning Tests
class TestProgressiveDifficulty:
    """Test progressive difficulty curriculum."""
    
    def test_initialization(self):
        """Test curriculum initialization."""
        curriculum = ProgressiveDifficulty(
            initial_difficulty=0.2,
            final_difficulty=0.8,
            num_steps=100
        )
        
        assert curriculum.initial_difficulty == 0.2
        assert curriculum.final_difficulty == 0.8
        assert curriculum.num_steps == 100
        assert curriculum.current_step == 0
    
    def test_threshold_progression(self):
        """Test difficulty threshold increases linearly."""
        curriculum = ProgressiveDifficulty(
            initial_difficulty=0.0,
            final_difficulty=1.0,
            num_steps=100
        )
        
        # At step 0
        threshold = curriculum.get_current_threshold()
        assert threshold == 0.0
        
        # At step 50
        curriculum.current_step = 50
        threshold = curriculum.get_current_threshold()
        assert abs(threshold - 0.5) < 1e-6
        
        # At step 100
        curriculum.current_step = 100
        threshold = curriculum.get_current_threshold()
        assert abs(threshold - 1.0) < 1e-6
    
    def test_example_selection(self):
        """Test selecting examples by difficulty."""
        curriculum = ProgressiveDifficulty(
            initial_difficulty=0.3,
            final_difficulty=0.7,
            num_steps=10
        )
        
        # Create dataset with difficulties
        dataset = [
            {'text': 'easy', 'difficulty': 0.2},
            {'text': 'medium', 'difficulty': 0.5},
            {'text': 'hard', 'difficulty': 0.8},
        ]
        
        # At start, only easy examples
        selected = curriculum.select_examples(dataset)
        assert len(selected) == 1
        assert selected[0]['text'] == 'easy'
        
        # Advance curriculum
        curriculum.update(accuracy=0.9, loss=0.1)
        curriculum.current_step = 5
        
        # Now medium difficulty allowed
        selected = curriculum.select_examples(dataset)
        assert len(selected) == 2


class TestCompetenceBasedPacing:
    """Test competence-based pacing curriculum."""
    
    def test_difficulty_adaptation(self):
        """Test difficulty adjusts based on performance."""
        curriculum = CompetenceBasedPacing(
            initial_difficulty=0.5,
            target_accuracy=0.8,
            difficulty_step=0.1
        )
        
        assert curriculum.current_difficulty == 0.5
        
        # High accuracy -> increase difficulty
        curriculum.update(accuracy=0.9, loss=0.1)
        assert curriculum.current_difficulty == 0.6
        
        # Low accuracy -> decrease difficulty
        curriculum.update(accuracy=0.6, loss=0.5)
        assert curriculum.current_difficulty == 0.5
    
    def test_should_advance(self):
        """Test advancement criteria."""
        curriculum = CompetenceBasedPacing(
            initial_difficulty=0.5,
            target_accuracy=0.8,
            patience=3
        )
        
        # Not ready yet
        assert not curriculum.should_advance()
        
        # Good performance for patience steps
        for _ in range(3):
            curriculum.update(accuracy=0.85, loss=0.1)
        
        # Now ready to advance
        assert curriculum.should_advance()


class TestMultiStageCurriculum:
    """Test multi-stage curriculum."""
    
    def test_stage_progression(self):
        """Test progression through stages."""
        stages = [
            CurriculumStage("easy", (0.0, 0.3), 0.7, 5, 1e-3),
            CurriculumStage("medium", (0.3, 0.7), 0.8, 10, 5e-4),
            CurriculumStage("hard", (0.7, 1.0), 0.85, 15, 1e-4),
        ]
        
        curriculum = MultiStageCurriculum(stages)
        
        # Start at easy stage
        assert curriculum.current_stage_idx == 0
        assert curriculum.get_current_stage().name == "easy"
        
        # High accuracy -> advance
        for _ in range(5):
            curriculum.update(accuracy=0.75, loss=0.1)
        
        # Should advance to medium
        assert curriculum.current_stage_idx == 1
        assert curriculum.get_current_stage().name == "medium"


class TestDifficultyScorer:
    """Test difficulty scoring."""
    
    def test_length_scoring(self):
        """Test length-based scoring."""
        scorer = DifficultyScorer()
        
        short_text = "Short"
        long_text = "This is a much longer text with many more words"
        
        short_score = scorer.score_length(short_text.split())
        long_score = scorer.score_length(long_text.split())
        
        assert long_score > short_score
    
    def test_vocabulary_scoring(self):
        """Test vocabulary-based scoring."""
        scorer = DifficultyScorer()
        
        # Update vocabulary with common words
        common = ["the", "a", "is", "in"]
        scorer.update_statistics([{"tokens": common}])
        
        # Rare words score higher
        rare_score = scorer.score_vocabulary(["quantum", "entanglement"])
        common_score = scorer.score_vocabulary(["the", "is"])
        
        assert rare_score > common_score


# Domain Adaptation Tests
class TestDomainAdapter:
    """Test domain adapter."""
    
    def test_freeze_embeddings(self):
        """Test freezing embedding layer."""
        model = SimpleModel()
        adapter = DomainAdapter(model)
        
        # Initially trainable
        assert model.embedding.weight.requires_grad
        
        # Freeze embeddings
        adapter.freeze_embeddings()
        assert not model.embedding.weight.requires_grad
    
    def test_freeze_encoder(self):
        """Test freezing encoder layers."""
        model = SimpleModel()
        adapter = DomainAdapter(model)
        
        # Freeze encoder
        adapter.freeze_encoder()
        
        # Check encoder params frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # Head still trainable
        assert model.head.weight.requires_grad
    
    def test_gradual_unfreeze(self):
        """Test gradual unfreezing."""
        model = SimpleModel()
        adapter = DomainAdapter(model)
        
        # Freeze all
        adapter.freeze_encoder()
        
        # Unfreeze top 1 layer
        adapter.gradual_unfreeze(num_layers=1)
        
        # Some params should be trainable now
        trainable = sum(p.requires_grad for p in model.parameters())
        assert trainable > 0


class TestFineTuner:
    """Test fine-tuner."""
    
    def test_progressive_strategy(self):
        """Test progressive fine-tuning."""
        model = SimpleModel()
        fine_tuner = FineTuner(
            model=model,
            strategy="progressive",
            unfreeze_schedule={1: 1, 5: 2}
        )
        
        # Setup freezes encoder
        fine_tuner.setup_progressive()
        
        # Check some params frozen
        frozen = sum(not p.requires_grad for p in model.parameters())
        assert frozen > 0
    
    def test_discriminative_strategy(self):
        """Test discriminative learning rates."""
        model = SimpleModel()
        fine_tuner = FineTuner(
            model=model,
            strategy="discriminative",
            layer_lr_decay=0.9
        )
        
        # Get parameter groups
        param_groups = fine_tuner.get_optimizer_params(base_lr=1e-4)
        
        # Should have multiple groups with different LRs
        assert len(param_groups) > 1


class TestDomainDiscriminator:
    """Test domain discriminator."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        discriminator = DomainDiscriminator(
            input_dim=64,
            hidden_dim=32,
            num_domains=3
        )
        
        features = torch.randn(4, 64)
        logits = discriminator(features)
        
        assert logits.shape == (4, 3)


class TestAdaptiveLayerNorm:
    """Test adaptive layer normalization."""
    
    def test_domain_specific_normalization(self):
        """Test domain-specific parameters."""
        norm = AdaptiveLayerNorm(dim=64, num_domains=3)
        
        x = torch.randn(4, 64)
        
        # Different domains produce different outputs
        out_domain0 = norm(x, domain_id=0)
        out_domain1 = norm(x, domain_id=1)
        
        assert not torch.allclose(out_domain0, out_domain1)


# Data Scheduling Tests
class TestDifficultyBasedSampler:
    """Test difficulty-based sampler."""
    
    def test_filtering_by_difficulty(self):
        """Test filtering examples by difficulty."""
        dataset = [
            {'text': 'easy', 'difficulty': 0.2},
            {'text': 'medium', 'difficulty': 0.5},
            {'text': 'hard', 'difficulty': 0.8},
        ]
        
        sampler = DifficultyBasedSampler(
            dataset=dataset,
            difficulty_range=(0.0, 0.6)
        )
        
        # Should select easy and medium
        assert len(list(iter(sampler))) == 2


class TestDynamicBatchComposer:
    """Test dynamic batch composer."""
    
    def test_batch_composition(self):
        """Test intelligent batch composition."""
        dataset = [
            {'text': 'short', 'difficulty': 0.3, 'domain': 'A'},
            {'text': 'medium text', 'difficulty': 0.5, 'domain': 'B'},
            {'text': 'very long text here', 'difficulty': 0.7, 'domain': 'A'},
        ]
        
        composer = DynamicBatchComposer(
            dataset=dataset,
            batch_size=2,
            difficulty_mode='mixed'
        )
        
        batch = composer.compose_batch()
        
        assert len(batch) == 2


class TestImportanceSampler:
    """Test importance sampler."""
    
    def test_weighted_sampling(self):
        """Test weighted sampling."""
        dataset = list(range(10))
        
        # High weight on first element
        weights = torch.zeros(10)
        weights[0] = 10.0
        
        sampler = ImportanceSampler(dataset, weights)
        
        # Sample multiple times
        samples = [next(iter(sampler)) for _ in range(100)]
        
        # First element should appear most often
        assert samples.count(0) > 50


# Optimization Tests
class TestWarmupScheduler:
    """Test warmup scheduler."""
    
    def test_warmup_progression(self):
        """Test LR increases during warmup."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = WarmupScheduler(optimizer, warmup_steps=10)
        
        # At step 0
        lrs = scheduler.get_lr()
        assert lrs[0] == 0.0
        
        # At step 5
        for _ in range(5):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(5e-5)
        
        # After warmup
        for _ in range(5):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs[0] == 1e-4


class TestCosineAnnealingScheduler:
    """Test cosine annealing scheduler."""
    
    def test_cosine_decay(self):
        """Test LR follows cosine curve."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingScheduler(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        initial_lr = scheduler.get_lr()[0]
        
        # Mid-point should be between min and max
        for _ in range(50):
            scheduler.step()
        
        mid_lr = scheduler.get_lr()[0]
        assert 1e-6 < mid_lr < 1e-4
        
        # End should approach min
        for _ in range(50):
            scheduler.step()
        
        final_lr = scheduler.get_lr()[0]
        assert final_lr == pytest.approx(1e-6, rel=1e-3)


class TestAdaptiveOptimizer:
    """Test adaptive optimizer."""
    
    def test_lr_reduction_on_plateau(self):
        """Test LR reduces when no improvement."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        adaptive_opt = AdaptiveOptimizer(
            optimizer,
            mode="min",
            patience=3,
            factor=0.5
        )
        
        initial_lr = adaptive_opt.get_lr()[0]
        
        # No improvement for patience steps
        for _ in range(3):
            adaptive_opt.step(metric=1.0)
        
        # LR should be reduced
        new_lr = adaptive_opt.get_lr()[0]
        assert new_lr == pytest.approx(initial_lr * 0.5)


# Multi-Task Learning Tests
class TestTaskHead:
    """Test task head."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        head = TaskHead(input_dim=64, output_dim=10, hidden_dim=32)
        
        x = torch.randn(4, 64)
        out = head(x)
        
        assert out.shape == (4, 10)


class TestMultiTaskModel:
    """Test multi-task model."""
    
    def test_single_task_forward(self):
        """Test forward pass for single task."""
        base_model = SimpleModel()
        shared_encoder = SharedEncoder(base_model, 64)
        
        task_heads = {
            'task_a': TaskHead(64, 5),
            'task_b': TaskHead(64, 3),
        }
        
        model = MultiTaskModel(shared_encoder, task_heads)
        
        x = torch.randint(0, 100, (4, 10))
        outputs = model(x, task='task_a')
        
        assert 'task_a' in outputs
        assert outputs['task_a'].shape == (4, 5)
    
    def test_multi_task_forward(self):
        """Test forward pass for all tasks."""
        base_model = SimpleModel()
        shared_encoder = SharedEncoder(base_model, 64)
        
        task_heads = {
            'task_a': TaskHead(64, 5),
            'task_b': TaskHead(64, 3),
        }
        
        model = MultiTaskModel(shared_encoder, task_heads)
        
        x = torch.randint(0, 100, (4, 10))
        outputs = model(x)
        
        assert 'task_a' in outputs
        assert 'task_b' in outputs


class TestTaskWeighting:
    """Test task weighting."""
    
    def test_equal_weighting(self):
        """Test equal weighting."""
        weighting = TaskWeighting(
            tasks=['task_a', 'task_b'],
            method='equal'
        )
        
        losses = {
            'task_a': torch.tensor(1.0),
            'task_b': torch.tensor(2.0),
        }
        
        total = weighting.compute_total_loss(losses)
        assert total == pytest.approx(3.0)


class TestTaskScheduler:
    """Test task scheduler."""
    
    def test_round_robin(self):
        """Test round-robin scheduling."""
        scheduler = TaskScheduler(
            tasks=['task_a', 'task_b', 'task_c'],
            method='round_robin'
        )
        
        # Should cycle through tasks
        assert scheduler.select_task() == 'task_a'
        assert scheduler.select_task() == 'task_b'
        assert scheduler.select_task() == 'task_c'
        assert scheduler.select_task() == 'task_a'


# Integration Tests
class TestIntegration:
    """Integration tests."""
    
    def test_curriculum_with_scheduler(self):
        """Test curriculum learning with data scheduler."""
        # Create curriculum
        curriculum = ProgressiveDifficulty(
            initial_difficulty=0.0,
            final_difficulty=1.0,
            num_steps=10
        )
        
        # Create dataset
        dataset = [
            {'text': f'example_{i}', 'difficulty': i * 0.1}
            for i in range(10)
        ]
        
        # Select examples
        selected = curriculum.select_examples(dataset)
        
        # Create sampler
        sampler = DifficultyBasedSampler(
            dataset=selected,
            difficulty_range=(0.0, 0.3)
        )
        
        assert len(list(iter(sampler))) > 0
    
    def test_domain_adaptation_with_optimizer(self):
        """Test domain adaptation with custom optimizer."""
        model = SimpleModel()
        
        # Setup domain adapter
        adapter = DomainAdapter(model)
        adapter.freeze_encoder()
        
        # Get parameter groups
        param_groups = adapter.get_parameter_groups(
            base_lr=1e-5,
            head_lr=1e-4
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups)
        
        assert len(optimizer.param_groups) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
