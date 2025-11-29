"""
NOVA Training Pipeline

Components:
- dataset.py: Dataset and DataLoader for training data
- corpus_processor.py: Convert raw text to AI2AI embeddings via Claude
- train_nova.py: Main training loop with next-token prediction
- validation.py: Validation metrics and evaluation
- checkpointing.py: Model saving/loading and training state
"""

from .dataset import NovaDataset, NovaDataLoader
from .corpus_processor import CorpusProcessor

__all__ = [
    "NovaDataset",
    "NovaDataLoader", 
    "CorpusProcessor",
]
