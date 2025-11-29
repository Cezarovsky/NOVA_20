# Data Pipeline Implementation

Complete data processing infrastructure for NOVA.

## Overview

The Data Pipeline provides comprehensive tools for:
- **Tokenization**: BPE-based tokenizer with domain-specific vocabulary
- **Dataset Loading**: Multiple strategies (memory, disk, streaming)
- **Preprocessing**: Text cleaning, normalization, chunking, quality filtering
- **Augmentation**: Data diversity through various transformation techniques
- **Batching**: Efficient collate functions with padding strategies

## Components

### 1. Tokenizer (`tokenizer.py`)

**NovaTokenizer** - BPE tokenizer with domain-specific vocabulary

```python
from src.data.tokenizer import NovaTokenizer

# Create tokenizer
tokenizer = NovaTokenizer(
    vocab_size=50000,
    min_frequency=2,
    add_domain_tokens=True  # Include physics, math, code symbols
)

# Train on corpus
texts = ["F = ma", "E = mc²", "∇·E = ρ/ε₀"]
tokenizer.train(texts, verbose=True)

# Encode text
text = "The force F equals mass times acceleration"
token_ids = tokenizer.encode(
    text,
    add_special_tokens=True,  # Add [BOS] and [EOS]
    max_length=512,
    truncation=True,
    padding=True
)

# Decode back
decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

# Save/load
tokenizer.save("tokenizer.json")
tokenizer = NovaTokenizer.load("tokenizer.json")
```

**Special Tokens**:
- `[PAD]` (0): Padding
- `[BOS]` (1): Beginning of sequence
- `[EOS]` (2): End of sequence
- `[UNK]` (3): Unknown token
- `[MASK]` (4): Masked token (for MLM)
- `[SEP]` (5): Separator
- `[CLS]` (6): Classification token

**Domain-Specific Tokens** (82 total):
- **Physics** (27): ∇, ∂, ∫, ∮, ∑, ∏, Greek letters (α-ω), constants (ℏ, ℓ, ∞, ∝)
- **Math** (24): √, ∛, ∜, ≈, ≠, ≤, ≥, ∈, ⊂, ∪, ∩, ∀, ∃, ∅, ℕ, ℤ, ℝ, ℂ, arrows
- **Code** (31): Python keywords (def, class, return, if, for, while, import, etc.)

### 2. Datasets (`datasets.py`)

#### TextDataset
Basic text dataset with on-the-fly tokenization.

```python
from src.data.datasets import TextDataset

# From list
texts = ["Sample 1", "Sample 2", "Sample 3"]
dataset = TextDataset(texts, tokenizer, max_length=512)

# From file
dataset = TextDataset("data.txt", tokenizer, max_length=512)

# Access examples
example = dataset[0]
# Returns: {'input_ids': [...], 'attention_mask': [...], 'text': "..."}
```

#### DomainDataset
Dataset with domain labels (physics, math, code, general).

```python
from src.data.datasets import DomainDataset

# From list of dicts
examples = [
    {"text": "F = ma", "domain": "physics"},
    {"text": "∫ x dx = x²/2", "domain": "math"},
    {"text": "def func(): return x", "domain": "code"},
]
dataset = DomainDataset(examples, tokenizer, max_length=512)

# From JSONL file
dataset = DomainDataset("data.jsonl", tokenizer)

# Filter by domain
physics_only = DomainDataset("data.jsonl", tokenizer, domain='physics')

# Access examples
example = dataset[0]
# Returns: {'input_ids': [...], 'attention_mask': [...], 
#           'domain_id': 0, 'domain': 'physics', 'text': "..."}
```

#### MultiDomainDataset
Balanced multi-domain sampling with mixing ratios.

```python
from src.data.datasets import MultiDomainDataset, DomainDataset

# Create domain datasets
physics_ds = DomainDataset("physics.jsonl", tokenizer)
math_ds = DomainDataset("math.jsonl", tokenizer)
code_ds = DomainDataset("code.jsonl", tokenizer)

# Combine with mixing ratios
dataset = MultiDomainDataset(
    domain_datasets={
        'physics': physics_ds,
        'math': math_ds,
        'code': code_ds,
    },
    mixing_ratios={
        'physics': 0.5,  # 50% physics
        'math': 0.3,     # 30% math
        'code': 0.2,     # 20% code
    },
    temperature=1.0  # 1.0 = as-is, <1.0 = more uniform
)

# Resample between epochs for variety
dataset.resample()
```

#### CachedDataset
Disk-cached preprocessing for faster training.

```python
from src.data.datasets import CachedDataset

# Wrap any dataset
cached_ds = CachedDataset(
    source_dataset=text_dataset,
    cache_file="cache.pkl",
    rebuild_cache=False  # Set True to force rebuild
)

# First run: preprocesses and saves to disk
# Subsequent runs: loads from cache (much faster)
```

#### MemoryMappedDataset
Memory mapping for very large datasets.

```python
from src.data.datasets import MemoryMappedDataset

# For datasets too large for RAM
dataset = MemoryMappedDataset(
    data_file="large_data.bin",
    index_file="large_data.idx",
    tokenizer=tokenizer,
    max_length=512
)

# Efficient random access without loading into memory
```

#### StreamingDataset
Infinite data generation for online learning.

```python
from src.data.datasets import StreamingDataset

def data_generator():
    """Generate infinite examples."""
    while True:
        yield generate_example()

dataset = StreamingDataset(
    data_generator=data_generator,
    tokenizer=tokenizer,
    max_length=512
)

# Use with DataLoader (no shuffle needed)
loader = DataLoader(dataset, batch_size=32)
```

### 3. Preprocessing (`preprocessing.py`)

#### TextPreprocessor
Complete preprocessing pipeline.

```python
from src.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=False,              # Keep case
    remove_urls=True,             # Remove URLs
    remove_emails=True,           # Remove emails
    normalize_unicode=True,       # Normalize unicode
    preserve_math_symbols=True,   # Keep ∇, ∂, etc.
)

text = "Visit https://example.com about ∇·E = ρ/ε₀"
processed = preprocessor(text)
# Result: "Visit about ∇·E = ρ/ε0"

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
processed_batch = preprocessor.preprocess_batch(texts)
```

#### DataCleaner
Quality-based filtering.

```python
from src.data.preprocessing import DataCleaner

cleaner = DataCleaner(
    min_length=10,                  # Min 10 chars
    max_length=10000,               # Max 10k chars
    max_repetition_ratio=0.5,       # Max 50% repetition
    min_entropy=3.0,                # Min character entropy
    max_special_char_ratio=0.3,     # Max 30% special chars
)

# Check validity
text = "This is good quality text"
if cleaner.is_valid(text):
    print("Valid!")

# Filter batch
texts = ["Good text", "abc", "aaa aaa aaa"]
valid_texts = cleaner.filter_batch(texts)
```

#### TextChunker
Split long texts into chunks.

```python
from src.data.preprocessing import TextChunker

chunker = TextChunker(
    chunk_size=512,           # Target chunk size
    overlap=50,               # Overlap between chunks
    respect_sentences=True,   # Chunk at sentence boundaries
)

long_text = "Very long document..."
chunks = chunker.chunk(long_text)
# Returns: list of text chunks
```

#### QualityFilter
Domain-specific quality scoring.

```python
from src.data.preprocessing import QualityFilter

# Physics quality filter
physics_filter = QualityFilter(domain='physics')

texts = [
    "F = ma with m = 10 kg and a = 2 m/s²",  # High score
    "Random text without physics",            # Low score
]

for text in texts:
    score = physics_filter.score(text)
    print(f"Score: {score:.2f}")

# Filter by minimum score
valid_texts = physics_filter.filter_batch(texts, min_score=0.5)
```

**Supported domains**: `physics`, `math`, `code`, `general`

### 4. Augmentation (`augmentation.py`)

#### SynonymReplacer
Replace words with synonyms.

```python
from src.data.augmentation import SynonymReplacer

augmentor = SynonymReplacer(
    augmentation_prob=0.5,      # 50% chance to augment
    replacement_prob=0.3,       # Replace 30% of words
)

text = "Calculate the energy of the system"
augmented = augmentor.augment(text)
# Possible: "Compute the power of the system"
```

#### ParaphraseAugmentor
Rule-based paraphrasing.

```python
from src.data.augmentation import ParaphraseAugmentor

augmentor = ParaphraseAugmentor(augmentation_prob=0.5)

text = "The velocity is equal to the derivative"
augmented = augmentor.augment(text)
# Possible: "The velocity equals the derivative"
```

#### BackTranslationAugmentor
Simulated back-translation.

```python
from src.data.augmentation import BackTranslationAugmentor

augmentor = BackTranslationAugmentor(
    augmentation_prob=0.5,
    intermediate_language='fr',
)

text = "Original sentence"
augmented = augmentor.augment(text)
```

#### DomainAugmentor
Domain-specific augmentations.

```python
from src.data.augmentation import DomainAugmentor

# Physics augmentation
physics_aug = DomainAugmentor(domain='physics', augmentation_prob=0.5)
text = "v = 10 m/s"
augmented = physics_aug.augment(text)
# Possible: "u = 10 m·s⁻¹"

# Math augmentation
math_aug = DomainAugmentor(domain='math', augmentation_prob=0.5)
text = "x >= y"
augmented = math_aug.augment(text)
# Possible: "x ≥ y"

# Code augmentation
code_aug = DomainAugmentor(domain='code', augmentation_prob=0.5)
```

#### Augmentation Pipeline
Combine multiple augmentors.

```python
from src.data.augmentation import create_augmentation_pipeline

pipeline = create_augmentation_pipeline(
    domain='physics',
    use_synonyms=True,
    use_paraphrase=True,
    use_back_translation=False,
    augmentation_prob=0.5,
)

text = "The energy is conserved"
augmented = pipeline.augment(text)
```

### 5. Collate Functions (`collate.py`)

#### Basic Collate
Simple padding to max length in batch.

```python
from torch.utils.data import DataLoader
from src.data.collate import collate_fn

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn,
)
```

#### Dynamic Padding
Pad only to max length in current batch.

```python
from src.data.collate import dynamic_padding_collate

collate = dynamic_padding_collate(
    pad_token_id=0,
    label_pad_token_id=-100,
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

#### Domain-Aware Collate
Add domain embeddings to batch.

```python
from src.data.collate import domain_aware_collate

collate = domain_aware_collate(
    pad_token_id=0,
    domain_to_id={'physics': 0, 'math': 1, 'code': 2, 'general': 3},
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate)

# Batch includes:
# - input_ids: [batch_size, seq_len]
# - attention_mask: [batch_size, seq_len]
# - domain_id: [batch_size]
# - domain_embedding: [batch_size, num_domains] (one-hot)
```

#### Variable-Length Collate
Bucket sequences by length.

```python
from src.data.collate import variable_length_collate

collate = variable_length_collate(
    pad_token_id=0,
    bucket_boundaries=[64, 128, 256, 512, 1024],
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

#### Sequence Packing
Pack multiple short sequences into one.

```python
from src.data.collate import sequence_packing_collate

collate = sequence_packing_collate(
    pad_token_id=0,
    max_length=512,
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

## Complete Pipeline Example

```python
from torch.utils.data import DataLoader
from src.data import (
    NovaTokenizer, DomainDataset, MultiDomainDataset,
    TextPreprocessor, DataCleaner,
    create_augmentation_pipeline,
    domain_aware_collate,
)

# 1. Train tokenizer
tokenizer = NovaTokenizer(vocab_size=50000)
tokenizer.train(all_texts, verbose=True)
tokenizer.save("tokenizer.json")

# 2. Preprocessing
preprocessor = TextPreprocessor(remove_urls=True, preserve_math_symbols=True)
cleaner = DataCleaner(min_length=10, max_length=1000)

raw_texts = load_raw_data()
processed_texts = preprocessor.preprocess_batch(raw_texts)
clean_texts = cleaner.filter_batch(processed_texts)

# 3. Create datasets
physics_ds = DomainDataset("physics.jsonl", tokenizer, domain='physics')
math_ds = DomainDataset("math.jsonl", tokenizer, domain='math')
code_ds = DomainDataset("code.jsonl", tokenizer, domain='code')

dataset = MultiDomainDataset(
    domain_datasets={'physics': physics_ds, 'math': math_ds, 'code': code_ds},
    mixing_ratios={'physics': 0.5, 'math': 0.3, 'code': 0.2},
)

# 4. Augmentation
augmentor = create_augmentation_pipeline(
    domain='physics',
    use_synonyms=True,
    use_paraphrase=True,
    augmentation_prob=0.3,
)

# 5. DataLoader
collate = domain_aware_collate(pad_token_id=tokenizer.pad_token_id)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate,
    num_workers=4,
)

# 6. Training loop
for batch in loader:
    input_ids = batch['input_ids']        # [32, seq_len]
    attention_mask = batch['attention_mask']
    domain_id = batch['domain_id']
    domain_embedding = batch['domain_embedding']
    
    # Forward pass
    outputs = model(input_ids, attention_mask, domain_embedding)
    loss = compute_loss(outputs, ...)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

## Performance Tips

### Memory Efficiency
1. **Use MemoryMappedDataset** for large datasets
2. **Enable sequence packing** to reduce padding waste
3. **Use dynamic padding** instead of fixed max_length
4. **Stream data** for infinite/online learning

### Speed Optimization
1. **Cache preprocessing** with CachedDataset
2. **Use multiple workers** in DataLoader
3. **Bucket by length** to minimize padding
4. **Preprocess offline** before training

### Quality Improvement
1. **Filter low-quality** with DataCleaner and QualityFilter
2. **Augment data** to increase diversity
3. **Balance domains** with MultiDomainDataset
4. **Domain-specific vocabulary** for specialized tasks

## Architecture

```
Data Pipeline
├── tokenizer.py (450 lines)
│   └── NovaTokenizer: BPE with domain vocabulary
├── datasets.py (440 lines)
│   ├── TextDataset: Basic loading
│   ├── DomainDataset: Domain labels
│   ├── MultiDomainDataset: Multi-domain sampling
│   ├── CachedDataset: Disk caching
│   ├── MemoryMappedDataset: Memory mapping
│   └── StreamingDataset: Infinite generation
├── preprocessing.py (430 lines)
│   ├── TextPreprocessor: Cleaning & normalization
│   ├── DataCleaner: Quality filtering
│   ├── TextChunker: Long text chunking
│   └── QualityFilter: Domain-specific scoring
├── augmentation.py (430 lines)
│   ├── SynonymReplacer: Synonym substitution
│   ├── ParaphraseAugmentor: Paraphrasing
│   ├── BackTranslationAugmentor: Back-translation
│   ├── DomainAugmentor: Domain-specific augmentation
│   └── CompositeAugmentor: Pipeline composition
└── collate.py (370 lines)
    ├── collate_fn: Basic padding
    ├── dynamic_padding_collate: Dynamic padding
    ├── domain_aware_collate: Domain embeddings
    ├── variable_length_collate: Bucketing
    └── sequence_packing_collate: Sequence packing

Total: ~2120 lines of production-ready data processing code
```

## Testing

Run comprehensive tests:
```bash
PYTHONPATH=. python examples/quick_data_test.py
```

All components tested:
- ✓ Tokenizer (train, encode, decode, save, load)
- ✓ Datasets (6 types, all loading strategies)
- ✓ Preprocessing (cleaning, chunking, filtering)
- ✓ Augmentation (5 techniques + pipeline)
- ✓ Collate (5 strategies + domain awareness)

## Integration with NOVA

The Data Pipeline integrates with:
- **ML Core**: Provides tokenized inputs for embedding layer
- **Training Pipeline**: DataLoader with collate functions
- **Advanced Training**: MultiDomainDataset for curriculum learning
- **Validation**: Quality metrics use domain-specific filtering

Perfect for:
- Multi-domain pretraining (physics, math, code)
- Domain adaptation with balanced sampling
- Curriculum learning with difficulty-based filtering
- Large-scale training with memory-efficient loading
