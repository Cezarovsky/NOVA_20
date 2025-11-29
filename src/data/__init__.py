"""
NOVA Data Pipeline

Complete data processing infrastructure:
- Tokenization (BPE with domain-specific tokens)
- Dataset classes (text, domain-specific, multi-domain)
- Preprocessing (cleaning, normalization, chunking)
- Augmentation (paraphrasing, back-translation)
- DataLoader utilities (collate, batching)
"""

from .tokenizer import NovaTokenizer
from .datasets import (
    TextDataset,
    DomainDataset,
    MultiDomainDataset,
    CachedDataset,
)
from .preprocessing import (
    TextPreprocessor,
    DataCleaner,
    TextChunker,
    QualityFilter,
)
from .augmentation import (
    DataAugmentor,
    ParaphraseAugmentor,
    BackTranslationAugmentor,
)
from .collate import (
    collate_fn,
    dynamic_padding_collate,
    domain_aware_collate,
)

__all__ = [
    # Tokenizer
    'NovaTokenizer',
    
    # Datasets
    'TextDataset',
    'DomainDataset',
    'MultiDomainDataset',
    'CachedDataset',
    
    # Preprocessing
    'TextPreprocessor',
    'DataCleaner',
    'TextChunker',
    'QualityFilter',
    
    # Augmentation
    'DataAugmentor',
    'ParaphraseAugmentor',
    'BackTranslationAugmentor',
    
    # Collate functions
    'collate_fn',
    'dynamic_padding_collate',
    'domain_aware_collate',
]
