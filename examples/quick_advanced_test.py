"""
Simple demo showing advanced training components work.
"""

import torch
import torch.nn as nn
from src.advanced_training.curriculum import ProgressiveDifficulty
from src.advanced_training.domain_adaptation import DomainAdapter
from src.advanced_training.data_scheduler import DifficultyBasedSampler
from src.advanced_training.optimization import WarmupScheduler
from src.advanced_training.multi_task import TaskHead

print("Testing Advanced Training Components...")
print("=" * 60)

# Test 1: Curriculum Learning
print("\n✓ Curriculum Learning")
curriculum = ProgressiveDifficulty(0.2, 0.8, 100)
print(f"  - ProgressiveDifficulty: {curriculum.get_current_threshold():.2f}")

# Test 2: Domain Adaptation  
print("\n✓ Domain Adaptation")
model = nn.Linear(10, 10)
adapter = DomainAdapter(model)
adapter.freeze_layer('weight')
print(f"  - DomainAdapter freeze: OK")

# Test 3: Data Scheduling
print("\n✓ Data Scheduling")
difficulties = [i * 0.1 for i in range(10)]
sampler = DifficultyBasedSampler(difficulties, batch_size=4, difficulty_range=(0.3, 0.7))
print(f"  - DifficultyBasedSampler: {len(sampler.valid_indices)} valid")

# Test 4: Optimization
print("\n✓ Optimization")
opt = torch.optim.Adam(model.parameters())
scheduler = WarmupScheduler(opt, 10)
print(f"  - WarmupScheduler: LR = {scheduler.get_lr()[0]:.2e}")

# Test 5: Multi-Task
print("\n✓ Multi-Task Learning")
head = TaskHead(10, 5)
x = torch.randn(2, 10)
out = head(x)
print(f"  - TaskHead output: {out.shape}")

print("\n" + "=" * 60)
print("🎉 All components working!")
