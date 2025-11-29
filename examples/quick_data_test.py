"""
Quick test for Data Pipeline components.

Verifies all components work correctly.
"""

import torch
from pathlib import Path
import tempfile

# Import data pipeline components
from src.data.tokenizer import NovaTokenizer
from src.data.datasets import (
    TextDataset, DomainDataset, MultiDomainDataset,
    CachedDataset, MemoryMappedDataset, StreamingDataset
)
from src.data.preprocessing import (
    TextPreprocessor, DataCleaner, TextChunker, QualityFilter
)
from src.data.augmentation import (
    SynonymReplacer, ParaphraseAugmentor, BackTranslationAugmentor,
    DomainAugmentor, create_augmentation_pipeline
)
from src.data.collate import (
    collate_fn, dynamic_padding_collate, domain_aware_collate,
    variable_length_collate, sequence_packing_collate
)


def test_tokenizer():
    """Test NovaTokenizer."""
    print("\n=== Testing Tokenizer ===")
    
    # Create tokenizer
    tokenizer = NovaTokenizer(vocab_size=1000, min_frequency=1)
    
    # Training texts
    texts = [
        "The force F = ma describes Newton's second law.",
        "Energy E = mc² is Einstein's famous equation.",
        "Velocity v = dx/dt is the derivative of position.",
        "∇²φ = 0 is Laplace's equation in physics.",
    ]
    
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer.train(texts, verbose=False)
    print(f"Vocabulary size: {tokenizer.vocab_length}")
    
    # Encode text
    text = "The force F equals mass times acceleration"
    encoded = tokenizer.encode(text, add_special_tokens=True, max_length=20, padding=True)
    print(f"Encoded: {encoded[:10]}... (showing first 10)")
    
    # Decode
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded: {decoded[:50]}...")
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tokenizer.json"
        tokenizer.save(save_path)
        print(f"Saved to {save_path}")
        
        loaded_tokenizer = NovaTokenizer.load(save_path)
        print(f"Loaded tokenizer with vocab size: {loaded_tokenizer.vocab_length}")
    
    print("✓ Tokenizer test passed")
    return tokenizer


def test_datasets(tokenizer):
    """Test dataset classes."""
    print("\n=== Testing Datasets ===")
    
    # Test TextDataset
    print("\n1. TextDataset")
    texts = [
        "This is a sample text.",
        "Another example for testing.",
        "Physics equations are interesting.",
    ]
    text_dataset = TextDataset(texts, tokenizer, max_length=20)
    print(f"TextDataset size: {len(text_dataset)}")
    sample = text_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {len(sample['input_ids'])}")
    
    # Test DomainDataset
    print("\n2. DomainDataset")
    domain_examples = [
        {"text": "F = ma is Newton's law", "domain": "physics"},
        {"text": "∫ x dx = x²/2", "domain": "math"},
        {"text": "def function(): return x", "domain": "code"},
    ]
    domain_dataset = DomainDataset(domain_examples, tokenizer, max_length=20)
    print(f"DomainDataset size: {len(domain_dataset)}")
    sample = domain_dataset[0]
    print(f"Sample domain: {sample['domain']}, domain_id: {sample['domain_id']}")
    
    # Test MultiDomainDataset
    print("\n3. MultiDomainDataset")
    physics_data = [{"text": "F = ma", "domain": "physics"}, {"text": "E = mc²", "domain": "physics"}]
    math_data = [{"text": "∫ x dx", "domain": "math"}, {"text": "d/dx x² = 2x", "domain": "math"}]
    
    physics_ds = DomainDataset(physics_data, tokenizer, max_length=20)
    math_ds = DomainDataset(math_data, tokenizer, max_length=20)
    
    multi_domain = MultiDomainDataset(
        domain_datasets={'physics': physics_ds, 'math': math_ds},
        mixing_ratios={'physics': 0.6, 'math': 0.4},
        temperature=1.0
    )
    print(f"MultiDomainDataset size: {len(multi_domain)}")
    sample = multi_domain[0]
    print(f"Sample domain: {sample['domain']}")
    
    # Test CachedDataset
    print("\n4. CachedDataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "cache.pkl"
        cached_dataset = CachedDataset(text_dataset, cache_file)
        print(f"CachedDataset size: {len(cached_dataset)}")
        sample = cached_dataset[0]
        print(f"Cached sample keys: {sample.keys()}")
    
    # Test StreamingDataset
    print("\n5. StreamingDataset")
    def data_generator():
        """Generate infinite data."""
        texts = ["Sample 1", "Sample 2", "Sample 3"]
        idx = 0
        while True:
            yield texts[idx % len(texts)]
            idx += 1
    
    streaming_ds = StreamingDataset(data_generator, tokenizer, max_length=20)
    print("StreamingDataset created (infinite)")
    iterator = iter(streaming_ds)
    sample = next(iterator)
    print(f"Streamed sample keys: {sample.keys()}")
    
    print("✓ All dataset tests passed")


def test_preprocessing():
    """Test preprocessing components."""
    print("\n=== Testing Preprocessing ===")
    
    # Test TextPreprocessor
    print("\n1. TextPreprocessor")
    preprocessor = TextPreprocessor(lowercase=False, remove_urls=True)
    text = "Visit https://example.com for more info about ∇·E = ρ/ε₀"
    processed = preprocessor(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    
    # Test DataCleaner
    print("\n2. DataCleaner")
    cleaner = DataCleaner(min_length=5, max_length=100)
    
    texts = [
        "Good text",  # Valid
        "abc",  # Too short
        "aaa aaa aaa aaa aaa aaa",  # High repetition
    ]
    
    for text in texts:
        is_valid = cleaner.is_valid(text)
        print(f"'{text[:30]}...' - Valid: {is_valid}")
    
    # Test TextChunker
    print("\n3. TextChunker")
    chunker = TextChunker(chunk_size=30, overlap=10, respect_sentences=True)
    long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = chunker.chunk(long_text)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0][:50]}...")
    
    # Test QualityFilter
    print("\n4. QualityFilter")
    physics_filter = QualityFilter(domain='physics')
    
    texts = [
        "The force F = ma with mass m = 10 kg and acceleration a = 2 m/s²",
        "This is just random text without physics content",
    ]
    
    for text in texts:
        score = physics_filter.score(text)
        print(f"Score: {score:.2f} - '{text[:40]}...'")
    
    print("✓ Preprocessing tests passed")


def test_augmentation():
    """Test augmentation components."""
    print("\n=== Testing Augmentation ===")
    
    text = "The energy of the system is conserved because of symmetry."
    
    # Test SynonymReplacer
    print("\n1. SynonymReplacer")
    synonym_aug = SynonymReplacer(augmentation_prob=1.0, replacement_prob=0.5)
    augmented = synonym_aug.augment(text)
    print(f"Original:  {text}")
    print(f"Augmented: {augmented}")
    
    # Test ParaphraseAugmentor
    print("\n2. ParaphraseAugmentor")
    paraphrase_aug = ParaphraseAugmentor(augmentation_prob=1.0)
    augmented = paraphrase_aug.augment(text)
    print(f"Paraphrased: {augmented}")
    
    # Test BackTranslationAugmentor
    print("\n3. BackTranslationAugmentor")
    back_trans_aug = BackTranslationAugmentor(augmentation_prob=1.0)
    augmented = back_trans_aug.augment(text)
    print(f"Back-translated: {augmented}")
    
    # Test DomainAugmentor
    print("\n4. DomainAugmentor")
    physics_aug = DomainAugmentor(domain='physics', augmentation_prob=1.0)
    physics_text = "The velocity v = 10 m/s"
    augmented = physics_aug.augment(physics_text)
    print(f"Physics augmented: {augmented}")
    
    # Test pipeline
    print("\n5. Augmentation Pipeline")
    pipeline = create_augmentation_pipeline(
        domain='physics',
        use_synonyms=True,
        use_paraphrase=True,
        augmentation_prob=0.8
    )
    augmented = pipeline.augment(text)
    print(f"Pipeline result: {augmented}")
    
    print("✓ Augmentation tests passed")


def test_collate(tokenizer):
    """Test collate functions."""
    print("\n=== Testing Collate Functions ===")
    
    # Create sample batch
    batch = [
        {'input_ids': [1, 2, 3, 4, 5], 'domain_id': 0, 'domain': 'physics'},
        {'input_ids': [1, 2, 3], 'domain_id': 1, 'domain': 'math'},
        {'input_ids': [1, 2, 3, 4, 5, 6, 7], 'domain_id': 0, 'domain': 'physics'},
    ]
    
    # Test basic collate_fn
    print("\n1. Basic collate_fn")
    collated = collate_fn(batch)
    print(f"Batch shape: {collated['input_ids'].shape}")
    print(f"Attention mask shape: {collated['attention_mask'].shape}")
    
    # Test dynamic_padding_collate
    print("\n2. Dynamic padding collate")
    dynamic_collate = dynamic_padding_collate(pad_token_id=0)
    collated = dynamic_collate(batch)
    print(f"Batch shape: {collated['input_ids'].shape}")
    
    # Test domain_aware_collate
    print("\n3. Domain-aware collate")
    domain_collate = domain_aware_collate(pad_token_id=0)
    collated = domain_collate(batch)
    print(f"Batch shape: {collated['input_ids'].shape}")
    print(f"Domain IDs: {collated['domain_id']}")
    print(f"Domain embeddings shape: {collated['domain_embedding'].shape}")
    
    # Test variable_length_collate
    print("\n4. Variable-length collate")
    var_collate = variable_length_collate(pad_token_id=0, bucket_boundaries=[8, 16, 32])
    collated = var_collate(batch)
    print(f"Bucketed batch shape: {collated['input_ids'].shape}")
    
    # Test sequence_packing_collate
    print("\n5. Sequence packing collate")
    packing_collate = sequence_packing_collate(pad_token_id=0, max_length=20)
    collated = packing_collate(batch)
    print(f"Packed batch shape: {collated['input_ids'].shape}")
    
    print("✓ Collate tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NOVA Data Pipeline - Quick Test")
    print("=" * 60)
    
    try:
        # Test tokenizer
        tokenizer = test_tokenizer()
        
        # Test datasets
        test_datasets(tokenizer)
        
        # Test preprocessing
        test_preprocessing()
        
        # Test augmentation
        test_augmentation()
        
        # Test collate functions
        test_collate(tokenizer)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
