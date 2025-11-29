"""
Advanced Training Demonstration

Demonstrates all advanced training capabilities:
1. Curriculum learning with progressive difficulty
2. Domain adaptation and fine-tuning
3. Advanced data scheduling
4. Optimization strategies (warmup, cosine annealing)
5. Multi-task learning
6. Full advanced training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

# NOVA imports
from src.advanced_training.curriculum import (
    ProgressiveDifficulty,
    CompetenceBasedPacing,
    MultiStageCurriculum,
    CurriculumStage,
    DifficultyScorer,
    create_curriculum,
)
from src.advanced_training.domain_adaptation import (
    DomainAdapter,
    FineTuner,
    DomainDiscriminator,
    AdaptiveLayerNorm,
)
from src.advanced_training.data_scheduler import (
    DifficultyBasedSampler,
    DynamicBatchComposer,
    StagedDataLoader,
    create_curriculum_dataloader,
)
from src.advanced_training.optimization import (
    WarmupScheduler,
    CosineAnnealingScheduler,
    WarmupCosineScheduler,
    AdaptiveOptimizer,
    DomainSpecificOptimizer,
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


# Simple model for demos
class DemoModel(nn.Module):
    """Simple transformer-like model for demos."""
    def __init__(self, vocab_size=1000, d_model=128, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
        for layer in self.encoder:
            x = layer(x)
        
        return self.output(x)  # [batch, seq_len, vocab]


def demo_curriculum_learning():
    """
    Demo 1: Curriculum Learning
    
    Shows progressive difficulty training with:
    - ProgressiveDifficulty strategy
    - CompetenceBasedPacing strategy
    - Multi-stage curriculum
    - Difficulty scoring
    """
    print("\n" + "="*60)
    print("DEMO 1: CURRICULUM LEARNING")
    print("="*60)
    
    # Create fake dataset with difficulties
    dataset = []
    for i in range(100):
        difficulty = np.random.rand()
        example = {
            'input_ids': torch.randint(0, 1000, (20,)),
            'text': f"Example {i}",
            'difficulty': difficulty,
        }
        dataset.append(example)
    
    print(f"\nDataset: {len(dataset)} examples")
    print(f"Difficulty range: {min(ex['difficulty'] for ex in dataset):.2f} - {max(ex['difficulty'] for ex in dataset):.2f}")
    
    # 1. Progressive Difficulty
    print("\n1. Progressive Difficulty Curriculum")
    print("-" * 40)
    
    curriculum = ProgressiveDifficulty(
        initial_difficulty=0.2,
        final_difficulty=0.8,
        num_steps=10,
        warmup_steps=2
    )
    
    print(f"Initial difficulty: {curriculum.initial_difficulty}")
    print(f"Final difficulty: {curriculum.final_difficulty}")
    print(f"Total steps: {curriculum.num_steps}")
    
    # Simulate training steps
    for step in range(5):
        threshold = curriculum.get_current_threshold()
        difficulties = [ex['difficulty'] for ex in dataset]
        model_performance = 0.8 + step * 0.05  # Increasing performance
        
        selected = curriculum.select_examples(dataset, difficulties, model_performance)
        
        print(f"\nStep {step}:")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Selected: {len(selected)}/{len(dataset)} examples")
        print(f"  Performance: {model_performance:.2f}")
        
        # Update curriculum
        curriculum.update(step=step, metrics={'accuracy': model_performance, 'loss': 1.0 - model_performance})
    
    # 2. Competence-Based Pacing
    print("\n2. Competence-Based Pacing")
    print("-" * 40)
    
    curriculum = CompetenceBasedPacing(
        target_accuracy=0.85,
        difficulty_step=0.05,
        patience=2
    )
    
    print(f"Initial difficulty: {curriculum.current_difficulty}")
    print(f"Target accuracy: {curriculum.target_accuracy}")
    
    # Simulate training with varying performance
    accuracies = [0.75, 0.80, 0.90, 0.85, 0.88]
    
    for step, acc in enumerate(accuracies):
        difficulties = [ex['difficulty'] for ex in dataset]
        selected = curriculum.select_examples(dataset, difficulties, acc)
        
        print(f"\nStep {step}:")
        print(f"  Accuracy: {acc:.2f}")
        print(f"  Current difficulty: {curriculum.current_difficulty:.2f}")
        print(f"  Selected: {len(selected)} examples")
        
        # Update curriculum
        curriculum.update(step=step, metrics={'accuracy': acc, 'loss': 1.0 - acc})
        
        if curriculum.should_advance():
            print("  ✓ Ready to advance!")
    
    # 3. Multi-Stage Curriculum
    print("\n3. Multi-Stage Curriculum")
    print("-" * 40)
    
    stages = [
        CurriculumStage("Easy", (0.0, 0.3), 0.75, 5, 1e-3),
        CurriculumStage("Medium", (0.3, 0.7), 0.85, 10, 5e-4),
        CurriculumStage("Hard", (0.7, 1.0), 0.90, 15, 1e-4),
    ]
    
    curriculum = MultiStageCurriculum(stages)
    
    print(f"Total stages: {len(stages)}")
    for i, stage in enumerate(stages):
        print(f"  Stage {i}: {stage.name}")
        print(f"    Difficulty: {stage.difficulty_range}")
        print(f"    Target accuracy: {stage.min_accuracy}")
    
    # Simulate stage progression
    print("\nTraining progression:")
    for epoch in range(15):
        stage = stages[curriculum.current_stage_idx]
        
        # Simulate accuracy improving over epochs
        acc = 0.65 + epoch * 0.02
        
        difficulties = [ex['difficulty'] for ex in dataset]
        selected = curriculum.select_examples(dataset, difficulties, acc)
        
        if epoch % 3 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Current stage: {stage.name}")
            print(f"  Accuracy: {acc:.2f}")
            print(f"  Selected: {len(selected)} examples")
        
        curriculum.update(step=epoch, metrics={'accuracy': acc, 'loss': 1.0 - acc})
    
    # 4. Difficulty Scoring
    print("\n4. Automatic Difficulty Scoring")
    print("-" * 40)
    
    scorer = DifficultyScorer(
        length_weight=0.3,
        vocab_weight=0.3,
        confidence_weight=0.4
    )
    
    # Update statistics from dataset
    scorer.update_statistics(dataset)
    
    print("Scoring examples:")
    test_examples = [
        {'input_ids': torch.randint(0, 1000, (10,)).tolist()},  # Short
        {'input_ids': torch.randint(0, 1000, (50,)).tolist()},  # Long
        {'input_ids': torch.randint(0, 100, (20,)).tolist()},   # Common vocab
    ]
    
    for i, example in enumerate(test_examples):
        score = scorer.score_example(example)
        print(f"  Example {i}: difficulty = {score:.3f}")
    
    print("\n✅ Curriculum learning demo complete!")


def demo_domain_adaptation():
    """
    Demo 2: Domain Adaptation
    
    Shows transfer learning with:
    - Freezing/unfreezing layers
    - Progressive fine-tuning
    - Discriminative learning rates
    - Domain discriminator
    - Adaptive layer normalization
    """
    print("\n" + "="*60)
    print("DEMO 2: DOMAIN ADAPTATION")
    print("="*60)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    
    # 1. Basic Domain Adapter
    print("\n1. Domain Adapter - Freeze/Unfreeze")
    print("-" * 40)
    
    adapter = DomainAdapter(model)
    
    print("Initial trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Freeze embeddings
    adapter.freeze_embeddings()
    print("After freezing embeddings:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Freeze encoder
    adapter.freeze_encoder()
    print("After freezing encoder:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Gradual unfreezing
    adapter.gradual_unfreeze(num_layers=2, reverse=True)
    print("After unfreezing top 2 layers:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # 2. Fine-Tuner with Progressive Strategy
    print("\n2. Progressive Fine-Tuning")
    print("-" * 40)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    
    fine_tuner = FineTuner(
        model=model,
        strategy="progressive"
    )
    
    # Setup progressive fine-tuning
    fine_tuner.setup_progressive()
    print("Initialized progressive fine-tuning")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Simulate epochs with unfreezing
    print("\nUnfreezing schedule:")
    for epoch in [1, 3, 5, 7]:
        fine_tuner.on_epoch_start(epoch)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Epoch {epoch}: {trainable} trainable parameters")
    
    # 3. Discriminative Learning Rates
    print("\n3. Discriminative Learning Rates")
    print("-" * 40)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    
    fine_tuner = FineTuner(
        model=model,
        strategy="discriminative"
    )
    
    param_groups = fine_tuner.get_optimizer_params(base_lr=1e-4)
    
    print(f"Created {len(param_groups)} parameter groups:")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: LR = {group['lr']:.2e}, {len(group['params'])} parameters")
    
    # 4. Domain Discriminator
    print("\n4. Domain Discriminator (Adversarial)")
    print("-" * 40)
    
    discriminator = DomainDiscriminator(
        input_dim=128,
        hidden_dim=64,
        num_domains=3
    )
    
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())} parameters")
    
    # Simulate features from different domains
    features = torch.randn(8, 128)
    domain_logits = discriminator(features)
    
    print(f"Input features: {features.shape}")
    print(f"Domain logits: {domain_logits.shape}")
    print(f"Predicted domains: {domain_logits.argmax(dim=1).tolist()}")
    
    # 5. Adaptive Layer Normalization
    print("\n5. Adaptive Layer Normalization")
    print("-" * 40)
    
    adaptive_norm = AdaptiveLayerNorm(
        normalized_shape=128,
        num_domains=3
    )
    
    x = torch.randn(4, 128)
    
    print("Same input, different domains:")
    for domain_id in range(3):
        output = adaptive_norm(x, domain_id=domain_id)
        print(f"  Domain {domain_id}: mean = {output.mean():.4f}, std = {output.std():.4f}")
    
    print("\n✅ Domain adaptation demo complete!")


def demo_data_scheduling():
    """
    Demo 3: Advanced Data Scheduling
    
    Shows intelligent batching with:
    - Difficulty-based sampling
    - Dynamic batch composition
    - Staged data loading
    - Importance sampling
    """
    print("\n" + "="*60)
    print("DEMO 3: ADVANCED DATA SCHEDULING")
    print("="*60)
    
    # Create dataset
    dataset = []
    for i in range(50):
        example = {
            'input_ids': torch.randint(0, 1000, (np.random.randint(10, 50),)),
            'difficulty': np.random.rand(),
            'domain': np.random.choice(['physics', 'math', 'code']),
        }
        dataset.append(example)
    
    print(f"Dataset: {len(dataset)} examples")
    
    # 1. Difficulty-Based Sampler
    print("\n1. Difficulty-Based Sampler")
    print("-" * 40)
    
    sampler = DifficultyBasedSampler(
        data_source=dataset,
        difficulty_key='difficulty',
        difficulty_range=(0.3, 0.7)
    )
    
    print(f"Difficulty range: {sampler.difficulty_range}")
    print(f"Valid indices: {len(sampler.valid_indices)}/{len(dataset)}")
    
    # Update difficulty range
    sampler.update_difficulty_range((0.5, 0.9))
    print(f"After update: {len(sampler.valid_indices)} valid examples")
    
    # 2. Dynamic Batch Composer
    print("\n2. Dynamic Batch Composition")
    print("-" * 40)
    
    composer = DynamicBatchComposer(
        data_source=dataset,
        batch_size=8,
        difficulty_mode='mixed',
        length_tolerance=10
    )
    
    print(f"Batch size: {composer.batch_size}")
    print(f"Difficulty mode: {composer.difficulty_mode}")
    print(f"Length tolerance: {composer.length_tolerance}")
    
    # Compose batch
    batch = composer.compose_batch()
    
    print(f"\nComposed batch:")
    print(f"  Size: {len(batch)}")
    print(f"  Difficulties: {[dataset[i]['difficulty'] for i in batch]}")
    print(f"  Lengths: {[len(dataset[i]['input_ids']) for i in batch]}")
    
    # 3. Staged Data Loading
    print("\n3. Staged Data Loading")
    print("-" * 40)
    
    # Split dataset into stages
    stage1_data = dataset[:20]
    stage2_data = dataset[20:40]
    stage3_data = dataset[40:]
    
    staged_loader = StagedDataLoader(
        datasets=[stage1_data, stage2_data, stage3_data],
        batch_sizes=[4, 8, 16]
    )
    
    print(f"Number of stages: {len(staged_loader.datasets)}")
    print(f"Current stage: {staged_loader.current_stage}")
    
    for stage in range(3):
        loader = staged_loader.get_current_loader()
        print(f"\nStage {stage}:")
        print(f"  Dataset size: {len(staged_loader.datasets[stage])}")
        print(f"  Batch size: {staged_loader.batch_sizes[stage]}")
        
        if stage < 2:
            staged_loader.advance_stage()
    
    # 4. Curriculum DataLoader Factory
    print("\n4. Curriculum DataLoader Factory")
    print("-" * 40)
    
    from torch.utils.data import TensorDataset
    
    # Create simple tensor dataset
    data = torch.randint(0, 1000, (100, 20))
    targets = torch.randint(0, 10, (100,))
    tensor_dataset = TensorDataset(data, targets)
    
    # Add difficulties
    for i, example in enumerate(tensor_dataset):
        tensor_dataset.tensors[0][i]  # Just accessing for consistency
    
    dataloader = create_curriculum_dataloader(
        dataset=tensor_dataset,
        batch_size=16,
        difficulty_range=(0.0, 1.0)
    )
    
    print(f"Created curriculum dataloader")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Num batches: {len(dataloader)}")
    
    print("\n✅ Data scheduling demo complete!")


def demo_optimization_strategies():
    """
    Demo 4: Optimization Strategies
    
    Shows advanced schedulers:
    - Warmup scheduler
    - Cosine annealing
    - Warmup + cosine
    - Adaptive optimizer
    - Domain-specific learning rates
    """
    print("\n" + "="*60)
    print("DEMO 4: OPTIMIZATION STRATEGIES")
    print("="*60)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    
    # 1. Warmup Scheduler
    print("\n1. Warmup Scheduler")
    print("-" * 40)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupScheduler(optimizer, warmup_steps=10)
    
    print(f"Warmup steps: 10")
    print(f"Target LR: 1e-4")
    print("\nLearning rate progression:")
    
    for step in [0, 2, 5, 10, 15]:
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        print(f"  Step {step:2d}: LR = {lr:.2e}")
    
    # 2. Cosine Annealing
    print("\n2. Cosine Annealing Scheduler")
    print("-" * 40)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingScheduler(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    print(f"T_max: 100 steps")
    print(f"Max LR: 1e-4")
    print(f"Min LR: 1e-6")
    print("\nLearning rate curve:")
    
    for step in [0, 25, 50, 75, 100]:
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        print(f"  Step {step:3d}: LR = {lr:.2e}")
    
    # 3. Warmup + Cosine
    print("\n3. Warmup + Cosine Scheduler")
    print("-" * 40)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=20,
        total_steps=100,
        min_lr=1e-6
    )
    
    print(f"Warmup: 20 steps")
    print(f"Total: 100 steps")
    print("\nCombined schedule:")
    
    for step in [0, 10, 20, 50, 100]:
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        phase = "Warmup" if step < 20 else "Cosine"
        print(f"  Step {step:3d} ({phase:7s}): LR = {lr:.2e}")
    
    # 4. Adaptive Optimizer
    print("\n4. Adaptive Optimizer")
    print("-" * 40)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    adaptive_opt = AdaptiveOptimizer(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
        verbose=False
    )
    
    print(f"Mode: minimize loss")
    print(f"Patience: 3 epochs")
    print(f"Reduction factor: 0.5")
    print("\nAdaptive LR adjustment:")
    
    # Simulate training with plateau
    losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.84]
    
    for epoch, loss in enumerate(losses):
        adaptive_opt.step(metric=loss)
        lr = adaptive_opt.get_lr()[0]
        print(f"  Epoch {epoch}: Loss = {loss:.2f}, LR = {lr:.2e}")
    
    # 5. Domain-Specific Optimizer
    print("\n5. Domain-Specific Optimizer")
    print("-" * 40)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    
    domain_lrs = {
        'embedding': 1e-5,
        'encoder': 5e-5,
        'output': 1e-4,
    }
    
    domain_opt = DomainSpecificOptimizer(
        model=model,
        domain_lrs=domain_lrs,
        default_lr=1e-4,
        optimizer_class=torch.optim.AdamW
    )
    
    print(f"Domain-specific learning rates:")
    for domain, lr in domain_lrs.items():
        print(f"  {domain}: {lr:.2e}")
    
    print(f"\nParameter groups: {len(domain_opt.optimizer.param_groups)}")
    
    print("\n✅ Optimization strategies demo complete!")


def demo_multi_task_learning():
    """
    Demo 5: Multi-Task Learning
    
    Shows multi-task training with:
    - Shared encoder + task heads
    - Task weighting strategies
    - Task scheduling
    - Multi-task trainer
    """
    print("\n" + "="*60)
    print("DEMO 5: MULTI-TASK LEARNING")
    print("="*60)
    
    # 1. Multi-Task Model
    print("\n1. Multi-Task Model Architecture")
    print("-" * 40)
    
    base_model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    shared_encoder = SharedEncoder(base_model, representation_dim=128)
    
    task_heads = {
        'sentiment': TaskHead(128, 3, hidden_dim=64),  # 3 classes
        'topic': TaskHead(128, 10, hidden_dim=64),     # 10 classes
        'language': TaskHead(128, 5, hidden_dim=64),   # 5 languages
    }
    
    multi_task_model = MultiTaskModel(shared_encoder, task_heads)
    
    print(f"Shared encoder parameters: {sum(p.numel() for p in shared_encoder.parameters())}")
    print(f"Task heads:")
    for name, head in task_heads.items():
        params = sum(p.numel() for p in head.parameters())
        print(f"  {name}: {params} parameters")
    
    # Test forward pass
    x = torch.randint(0, 1000, (4, 20))
    outputs = multi_task_model(x)
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    for task, output in outputs.items():
        print(f"  {task} output: {output.shape}")
    
    # 2. Task Weighting
    print("\n2. Task Weighting Strategies")
    print("-" * 40)
    
    tasks = ['sentiment', 'topic', 'language']
    
    # Equal weighting
    weighting_equal = TaskWeighting(tasks, method="equal")
    print("Equal weighting:")
    
    losses = {
        'sentiment': torch.tensor(0.5),
        'topic': torch.tensor(1.0),
        'language': torch.tensor(0.8),
    }
    
    weights = weighting_equal.get_weights(losses)
    for task, weight in weights.items():
        print(f"  {task}: {weight:.2f}")
    
    total_loss = weighting_equal.compute_total_loss(losses)
    print(f"  Total loss: {total_loss:.2f}")
    
    # Uncertainty weighting
    print("\nUncertainty weighting:")
    weighting_uncertainty = TaskWeighting(tasks, method="uncertainty")
    
    # Simulate multiple steps
    for step in range(3):
        weights = weighting_uncertainty.get_weights(losses)
        total_loss = weighting_uncertainty.compute_total_loss(losses)
        
        if step == 0:
            for task, weight in weights.items():
                print(f"  {task}: weight = {weight:.2f}")
    
    # 3. Task Scheduling
    print("\n3. Task Scheduling")
    print("-" * 40)
    
    task_sizes = {
        'sentiment': 1000,
        'topic': 500,
        'language': 2000,
    }
    
    # Round-robin
    scheduler_rr = TaskScheduler(tasks, method="round_robin")
    print("Round-robin scheduling:")
    print("  ", [scheduler_rr.select_task() for _ in range(6)])
    
    # Proportional
    scheduler_prop = TaskScheduler(tasks, method="proportional", task_sizes=task_sizes)
    print("\nProportional scheduling (by dataset size):")
    
    # Count selections
    selections = [scheduler_prop.select_task() for _ in range(100)]
    for task in tasks:
        count = selections.count(task)
        print(f"  {task}: {count}% ({task_sizes[task]} examples)")
    
    # 4. Complete Multi-Task Training Example
    print("\n4. Multi-Task Training Example")
    print("-" * 40)
    
    # Create simple tasks
    task_configs = [
        Task("sentiment", num_classes=3, weight=1.0, loss_fn=nn.CrossEntropyLoss()),
        Task("topic", num_classes=10, weight=1.5, loss_fn=nn.CrossEntropyLoss()),
    ]
    
    # Build model
    base_model = DemoModel(vocab_size=1000, d_model=128, num_layers=2)
    multi_task_model = create_multi_task_model(
        base_model=base_model,
        task_configs=task_configs,
        shared_dim=128,
        hidden_dim=64
    )
    
    print("Created multi-task model:")
    print(f"  Tasks: {list(multi_task_model.task_heads.keys())}")
    print(f"  Total parameters: {sum(p.numel() for p in multi_task_model.parameters())}")
    
    # Simulate training step
    print("\nSimulated training step:")
    
    # Fake batches
    x = torch.randint(0, 1000, (8, 20))
    y_sentiment = torch.randint(0, 3, (8,))
    y_topic = torch.randint(0, 10, (8,))
    
    # Forward pass
    outputs = multi_task_model(x)
    
    # Compute losses
    loss_sentiment = task_configs[0].loss_fn(
        outputs['sentiment'].mean(dim=1),  # Pool sequence
        y_sentiment
    )
    loss_topic = task_configs[1].loss_fn(
        outputs['topic'].mean(dim=1),
        y_topic
    )
    
    print(f"  Sentiment loss: {loss_sentiment:.3f}")
    print(f"  Topic loss: {loss_topic:.3f}")
    print(f"  Combined: {loss_sentiment + loss_topic:.3f}")
    
    print("\n✅ Multi-task learning demo complete!")


def demo_full_pipeline():
    """
    Demo 6: Full Advanced Training Pipeline
    
    Combines all techniques:
    - Curriculum learning
    - Domain adaptation
    - Multi-task learning
    - Advanced optimization
    """
    print("\n" + "="*60)
    print("DEMO 6: FULL ADVANCED TRAINING PIPELINE")
    print("="*60)
    
    print("\nBuilding comprehensive training setup...")
    
    # 1. Model with domain adaptation
    print("\n1. Model Setup")
    print("-" * 40)
    
    model = DemoModel(vocab_size=1000, d_model=128, num_layers=4)
    print(f"Base model: {sum(p.numel() for p in model.parameters())} parameters")
    
    adapter = DomainAdapter(model)
    adapter.freeze_embeddings()
    print("Domain adapter: Froze embeddings")
    
    # 2. Curriculum
    print("\n2. Curriculum Learning")
    print("-" * 40)
    
    curriculum = ProgressiveDifficulty(
        initial_difficulty=0.2,
        final_difficulty=0.9,
        num_steps=100
    )
    print(f"Curriculum: Progressive difficulty (0.2 → 0.9 over 100 steps)")
    
    # 3. Optimizer with schedule
    print("\n3. Optimization")
    print("-" * 40)
    
    optimizer, scheduler = create_optimizer_with_schedule(
        model=model,
        optimizer_name="adamw",
        lr=1e-4,
        schedule="warmup_cosine",
        warmup_steps=1000,
        total_steps=10000
    )
    print(f"Optimizer: AdamW")
    print(f"Schedule: Warmup (1000) + Cosine Annealing (10000)")
    
    # 4. Simulate training loop
    print("\n4. Training Simulation")
    print("-" * 40)
    
    print("Running 10 training steps...\n")
    
    for step in range(10):
        # Get curriculum threshold
        threshold = curriculum.get_current_threshold()
        
        # Get learning rate
        lr = scheduler.get_lr()[0]
        
        # Simulate training
        model.train()
        x = torch.randint(0, 1000, (16, 20))
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(
            output.reshape(-1, 1000),
            x.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update curriculum
        acc = 0.6 + step * 0.03
        curriculum.update(step=step, metrics={'accuracy': acc, 'loss': loss.item()})
        
        if step % 3 == 0:
            print(f"Step {step:2d}:")
            print(f"  Loss: {loss.item():.3f}")
            print(f"  Accuracy: {acc:.2f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Curriculum threshold: {threshold:.2f}")
    
    print("\n✅ Full pipeline demo complete!")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("NOVA ADVANCED TRAINING DEMONSTRATIONS")
    print("="*60)
    print("\nDemonstrating all advanced training capabilities:")
    print("  1. Curriculum Learning")
    print("  2. Domain Adaptation")
    print("  3. Data Scheduling")
    print("  4. Optimization Strategies")
    print("  5. Multi-Task Learning")
    print("  6. Full Training Pipeline")
    
    # Run all demos
    demo_curriculum_learning()
    demo_domain_adaptation()
    demo_data_scheduling()
    demo_optimization_strategies()
    demo_multi_task_learning()
    demo_full_pipeline()
    
    print("\n" + "="*60)
    print("🎉 ALL DEMOS COMPLETE!")
    print("="*60)
    print("\nAdvanced training system ready for production use.")
    print("See ADVANCED_TRAINING_IMPLEMENTATION.md for documentation.")


if __name__ == "__main__":
    main()
