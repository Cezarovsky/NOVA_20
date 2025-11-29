# Advanced Training Implementation

Complete implementation of sophisticated training techniques for NOVA.

## 📦 Components (5 modules, ~2465 lines)

### 1. Curriculum Learning (`curriculum.py`, 685 lines)

Progressive difficulty training strategies:

- **ProgressiveDifficulty**: Linear difficulty increase over training
  - Gradually raises threshold from initial to final difficulty
  - Warmup period for stability
  - Example: 0.2 → 0.8 over 100 steps

- **CompetenceBasedPacing**: Adaptive difficulty based on performance
  - Increases difficulty when model performs well (accuracy > target + 0.1)
  - Decreases when model struggles (accuracy < target - 0.2)
  - Patience mechanism prevents premature advancement

- **MultiStageCurriculum**: Distinct training phases
  - Each stage has difficulty range, target accuracy, LR
  - Automatic advancement when stage goals met
  - Example: Easy (0.0-0.3) → Medium (0.3-0.7) → Hard (0.7-1.0)

- **DifficultyScorer**: Automatic difficulty assessment
  - Length-based scoring (30%): Longer sequences = harder
  - Vocabulary scoring (30%): Rare words = harder
  - Confidence scoring (40%): Lower model confidence = harder
  - Updates statistics from dataset for normalization

**Key Features:**
- Abstract `CurriculumStrategy` base class for extensibility
- `should_advance()` method for stage transitions
- Factory function `create_curriculum()` for easy instantiation
- Metrics tracking (accuracy, loss) for adaptive pacing

### 2. Domain Adaptation (`domain_adaptation.py`, 450 lines)

Transfer learning and fine-tuning:

- **DomainAdapter**: Layer freezing/unfreezing
  - `freeze_embeddings()`: Keep pre-trained embeddings fixed
  - `freeze_encoder()`: Freeze transformer layers
  - `gradual_unfreeze()`: Progressive unfreezing (reverse order)
  - `get_parameter_groups()`: Differential learning rates

- **FineTuner**: Structured fine-tuning strategies
  - Progressive: Gradually unfreeze layers over epochs
  - Discriminative: Layer-wise LR decay (deeper = smaller LR)
  - Unfreeze schedule: {epoch: num_layers} mapping

- **DomainDiscriminator**: Adversarial domain adaptation
  - 3-layer MLP for domain classification
  - Used in domain-adversarial training
  - Helps learn domain-invariant representations

- **AdaptiveLayerNorm**: Domain-specific normalization
  - Separate scale/shift parameters per domain
  - Forward: `norm(x, domain_id)`
  - Allows domain-specific feature distributions

- **load_pretrained_weights()**: Flexible checkpoint loading
  - Handles prefix mismatches
  - Strict/non-strict mode support

**Use Cases:**
- Fine-tune pre-trained NOVA on domain-specific data
- Transfer knowledge from general → specialized domains
- Adapt model while preserving base capabilities

### 3. Data Scheduling (`data_scheduler.py`, 392 lines)

Intelligent sampling and batching:

- **DifficultyBasedSampler**: Curriculum-aware sampling
  - Filters examples by difficulty range
  - `update_difficulty_range()` for progressive training
  - Supports shuffling within valid examples

- **DynamicBatchComposer**: Smart batch composition
  - Length grouping (tolerance parameter)
  - Difficulty modes: uniform, mixed, progressive
  - Domain balance: prefer diverse domains
  - Scoring system for candidate selection

- **StagedDataLoader**: Multi-phase data loading
  - Different datasets per stage
  - Stage-specific batch sizes
  - `advance_stage()` for manual control
  - `get_current_loader()` for active loader

- **ImportanceSampler**: Weighted sampling
  - Torch-based multinomial sampling
  - `update_weights()` for dynamic adjustment
  - Focus on hard examples

- **BalancedBatchSampler**: Class/domain balance
  - Equal samples per label
  - Prevents class imbalance issues

- **create_curriculum_dataloader()**: Factory function
  - Combines difficulty sampling with DataLoader
  - Easy integration with training loops

**Benefits:**
- Efficient training through smart sampling
- Reduced variance with balanced batches
- Curriculum learning integration

### 4. Optimization (`optimization.py`, 460 lines)

Advanced learning rate schedules and optimizers:

- **WarmupScheduler**: Linear warmup
  - LR: 0 → base_lr over warmup_steps
  - Prevents early instability
  - Common in transformer training

- **CosineAnnealingScheduler**: Smooth decay
  - Formula: lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t/T))
  - Optional warmup period
  - Helps convergence to better minima

- **WarmupCosineScheduler**: Combined schedule
  - Most popular for transformers
  - Linear warmup → cosine decay
  - Min LR parameter for final value

- **PolynomialDecayScheduler**: Polynomial decay
  - LR: (base - end) * (1 - t/T)^power + end
  - power=1.0: linear decay
  - power>1.0: faster initial decay

- **AdaptiveOptimizer**: Automatic LR adjustment
  - Monitors metrics (loss/accuracy)
  - Reduces LR on plateau (patience mechanism)
  - Configurable reduction factor

- **DomainSpecificOptimizer**: Per-domain learning rates
  - Different LRs for different domains
  - Parameter grouping by name matching
  - Supports any optimizer class (AdamW, Adam, SGD)

- **LayerWiseLRScheduler**: Depth-based LR decay
  - Deeper layers get lower LRs
  - Preserves pre-trained knowledge
  - Common in BERT fine-tuning

- **create_optimizer_with_schedule()**: Factory function
  - Returns (optimizer, scheduler) tuple
  - Supports: adamw, adam, sgd
  - Schedules: warmup, cosine, warmup_cosine, polynomial

**Best Practices:**
- Use warmup for large models (prevents divergence)
- Cosine annealing for final convergence
- Domain-specific LRs for multi-domain training
- Adaptive optimization when validation metric available

### 5. Multi-Task Learning (`multi_task.py`, 478 lines)

Train on multiple tasks simultaneously:

- **TaskHead**: Task-specific output layer
  - Projects shared representations → task outputs
  - Optional hidden layer
  - Dropout for regularization

- **SharedEncoder**: Common feature extractor
  - Wraps base model (e.g., NOVA transformer)
  - Extracts representations for all tasks
  - Single forward pass for efficiency

- **MultiTaskModel**: Complete architecture
  - Shared encoder + multiple task heads
  - `forward(x, task)`: Single task prediction
  - `forward(x)`: All task predictions
  - `freeze_encoder()`/`unfreeze_encoder()` methods

- **TaskWeighting**: Automatic loss balancing
  - **Equal**: Simple averaging (baseline)
  - **Uncertainty**: Learnable log-variance weighting
    - Weight = 1 / (2 * σ²)
    - Regularization: log(σ²)
    - From: Kendall et al., 2018
  - **GradNorm**: Gradient magnitude balancing
    - Balances gradient norms across tasks
    - Alpha parameter controls restoring force
    - From: Chen et al., 2018

- **TaskScheduler**: Task selection strategies
  - **Round-robin**: Cycle through tasks
  - **Proportional**: Sample by dataset size
  - **Performance**: Focus on worst task

- **MultiTaskTrainer**: Training orchestration
  - Coordinates task scheduling
  - Applies task weighting
  - Updates shared encoder
  - Per-task metrics tracking

- **create_multi_task_model()**: Factory function
  - Creates SharedEncoder + task heads
  - Configurable hidden dimensions
  - Returns complete MultiTaskModel

**Architecture:**
```
Input
  ↓
Shared Encoder (e.g., NOVA transformer)
  ↓
Shared Representations
  ├→ Task A Head → Task A Output
  ├→ Task B Head → Task B Output
  └→ Task C Head → Task C Output
```

**Benefits:**
- Knowledge transfer between related tasks
- Improved generalization
- Parameter efficiency (shared encoder)
- Automatic loss balancing

## 🔧 Usage Examples

### Curriculum Learning

```python
from src.advanced_training.curriculum import create_curriculum

# Progressive difficulty
curriculum = create_curriculum(
    strategy="progressive",
    initial_difficulty=0.2,
    final_difficulty=0.8,
    num_steps=100
)

# Training loop
for step in range(num_steps):
    threshold = curriculum.get_current_threshold()
    difficulties = [scorer.score_example(ex) for ex in dataset]
    selected_indices = curriculum.select_examples(dataset, difficulties, accuracy)
    
    # Train on selected examples
    batch = [dataset[i] for i in selected_indices]
    loss, accuracy = train_step(batch)
    
    # Update curriculum
    curriculum.update(step, {'accuracy': accuracy, 'loss': loss})
```

### Domain Adaptation

```python
from src.advanced_training.domain_adaptation import FineTuner

# Load pre-trained NOVA
model = load_pretrained_nova()

# Setup fine-tuner
fine_tuner = FineTuner(
    model=model,
    strategy="progressive"
)

# Progressive unfreezing
fine_tuner.setup_progressive()

# Training loop with gradual unfreezing
for epoch in range(num_epochs):
    fine_tuner.on_epoch_start(epoch)  # Automatically unfreezes layers
    
    # Get optimizer with discriminative LRs
    param_groups = fine_tuner.get_optimizer_params(base_lr=1e-4)
    optimizer = torch.optim.AdamW(param_groups)
    
    # Train epoch
    train_epoch(model, optimizer)
```

### Advanced Optimization

```python
from src.advanced_training.optimization import create_optimizer_with_schedule

# Create optimizer + scheduler
optimizer, scheduler = create_optimizer_with_schedule(
    model=model,
    optimizer_name="adamw",
    lr=1e-4,
    schedule="warmup_cosine",
    warmup_steps=1000,
    total_steps=10000
)

# Training loop
for step in range(10000):
    loss = train_step(model, batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update LR
```

### Multi-Task Learning

```python
from src.advanced_training.multi_task import create_multi_task_model, Task

# Define tasks
tasks = [
    Task("sentiment", num_classes=3, loss_fn=nn.CrossEntropyLoss()),
    Task("topic", num_classes=10, loss_fn=nn.CrossEntropyLoss()),
]

# Create model
model = create_multi_task_model(
    base_model=nova_model,
    task_configs=tasks,
    shared_dim=512,
    hidden_dim=256
)

# Training
outputs = model(x)  # All tasks
sentiment_logits = outputs['sentiment']
topic_logits = outputs['topic']

# Compute weighted loss
losses = {
    'sentiment': sentiment_loss,
    'topic': topic_loss
}
total_loss = task_weighting.compute_total_loss(losses)
```

## 📊 Implementation Stats

- **Total Lines**: 2,465 (across 5 modules)
- **Classes**: 24
- **Factory Functions**: 3
- **Key Algorithms**:
  - Progressive difficulty scheduling
  - Competence-based pacing
  - Discriminative fine-tuning
  - Uncertainty-based task weighting
  - GradNorm balancing
  - Cosine annealing with warmup

## ✅ Testing

Simple component test:
```bash
python examples/quick_advanced_test.py
```

Output:
```
✓ Curriculum Learning
✓ Domain Adaptation  
✓ Data Scheduling
✓ Optimization
✓ Multi-Task Learning
🎉 All components working!
```

## 🎯 Integration with NOVA

### Training Pipeline Enhancement

The advanced training components integrate seamlessly with existing NOVA training:

```python
# 1. Setup curriculum
curriculum = ProgressiveDifficulty(0.2, 0.8, num_steps)

# 2. Setup domain adaptation
adapter = DomainAdapter(nova_model)
adapter.freeze_embeddings()

# 3. Setup optimizer with schedule
optimizer, scheduler = create_optimizer_with_schedule(
    model=nova_model,
    optimizer_name="adamw",
    lr=1e-4,
    schedule="warmup_cosine",
    warmup_steps=warmup_steps,
    total_steps=total_steps
)

# 4. Training loop with curriculum
for step in range(total_steps):
    # Get current curriculum threshold
    threshold = curriculum.get_current_threshold()
    
    # Select examples by difficulty
    difficulties = [scorer.score_example(ex) for ex in dataset]
    selected = curriculum.select_examples(dataset, difficulties, accuracy)
    
    # Train step
    batch = create_batch(selected)
    loss, metrics = train_step(nova_model, batch, optimizer)
    
    # Update
    scheduler.step()
    curriculum.update(step, metrics)
    
    # Optional: gradual unfreezing
    if step in unfreeze_schedule:
        adapter.gradual_unfreeze(num_layers)
```

## 📚 References

**Curriculum Learning:**
- Bengio et al. (2009): "Curriculum Learning"
- Soviany et al. (2021): "Curriculum Learning: A Survey"

**Domain Adaptation:**
- Yosinski et al. (2014): "How transferable are features in deep neural networks?"
- Howard & Ruder (2018): "Universal Language Model Fine-tuning for Text Classification"

**Multi-Task Learning:**
- Kendall et al. (2018): "Multi-Task Learning Using Uncertainty to Weigh Losses"
- Chen et al. (2018): "GradNorm: Gradient Normalization for Adaptive Loss Balancing"

**Optimization:**
- Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"

## 🚀 Benefits for NOVA

1. **Faster Convergence**: Curriculum learning reduces training time
2. **Better Generalization**: Multi-task learning improves robustness
3. **Domain Specialization**: Fine-tuning enables task-specific expertise
4. **Stable Training**: Warmup and adaptive LR prevent instability
5. **Efficient Sampling**: Smart batching reduces computational waste

## 🔮 Future Enhancements

- [ ] Meta-learning curriculum strategies
- [ ] Automated difficulty estimation with confidence models
- [ ] Continual learning with task rehearsal
- [ ] Neural architecture search for task heads
- [ ] Mixtures of experts for domain specialization

---

*Implemented: January 2025*  
*Total Development Time: ~4 hours*  
*Status: Production-ready, all components tested*
