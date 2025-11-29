"""
NOVA Validation & Metrics

Comprehensive evaluation metrics for NOVA training and inference:
- Language modeling: perplexity, accuracy, loss
- Generation quality: BLEU, ROUGE, METEOR
- Embedding similarity: cosine, L2 distance
- Domain-specific: custom evaluators
"""

from .metrics import (
    Perplexity,
    Accuracy,
    TokenAccuracy,
    BLEU,
    ROUGE,
    EmbeddingSimilarity,
    MetricsTracker,
)
from .validators import (
    ModelValidator,
    EmbeddingValidator,
    GenerationValidator,
)
from .benchmarks import (
    LanguageModelingBenchmark,
    DomainBenchmark,
    BenchmarkSuite,
)

__all__ = [
    # Core metrics
    "Perplexity",
    "Accuracy",
    "TokenAccuracy",
    # Generation metrics
    "BLEU",
    "ROUGE",
    # Embedding metrics
    "EmbeddingSimilarity",
    # Tracking
    "MetricsTracker",
    # Validators
    "ModelValidator",
    "EmbeddingValidator",
    "GenerationValidator",
    # Benchmarks
    "LanguageModelingBenchmark",
    "DomainBenchmark",
    "BenchmarkSuite",
]
