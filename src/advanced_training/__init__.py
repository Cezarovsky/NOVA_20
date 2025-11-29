"""
NOVA Advanced Training

Sophisticated training techniques for optimal model performance:
- Curriculum learning: Progressive difficulty training
- Domain adaptation: Transfer learning and specialization
- Multi-task learning: Shared representations
- Advanced optimization: Adaptive learning strategies
"""

from .curriculum import (
    CurriculumStrategy,
    ProgressiveDifficulty,
    CompetenceBasedPacing,
    MultiStageCurriculum,
    DifficultyScorer,
)
from .domain_adaptation import (
    DomainAdapter,
    FineTuner,
    DomainDiscriminator,
    AdaptiveLayerNorm,
)
from .data_scheduler import (
    DifficultyBasedSampler,
    DynamicBatchComposer,
    StagedDataLoader,
    ImportanceSampler,
    BalancedBatchSampler,
    create_curriculum_dataloader,
)
from .multi_task import (
    MultiTaskTrainer,
    TaskHead,
    TaskWeighting,
    SharedEncoder,
)
from .optimization import (
    AdaptiveOptimizer,
    WarmupScheduler,
    CosineAnnealingScheduler,
    DomainSpecificOptimizer,
)

__all__ = [
    # Curriculum learning
    "CurriculumStrategy",
    "ProgressiveDifficulty",
    "CompetenceBasedPacing",
    "MultiStageCurriculum",
    "DifficultyScorer",
    # Domain adaptation
    "DomainAdapter",
    "FineTuner",
    "DomainDiscriminator",
    "AdaptiveLayerNorm",
    # Data scheduling
    "DataScheduler",
    "DifficultyBasedSampler",
    "DynamicBatchComposer",
    "StagedDataLoader",
    # Multi-task learning
    "MultiTaskTrainer",
    "TaskHead",
    "TaskWeighting",
    "SharedEncoder",
    # Optimization
    "AdaptiveOptimizer",
    "WarmupScheduler",
    "CosineAnnealingScheduler",
    "DomainSpecificOptimizer",
]
