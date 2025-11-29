"""
NOVA Training Demo

Demonstrates complete training pipeline:
1. Load/create corpus
2. Process with Claude (ONE-TIME) → AI2AI embeddings
3. Train NOVA with embeddings
4. NOVA operates independently!
"""

import torch
from pathlib import Path

from src.ml.transformer import Transformer
from src.training.dataset import NovaDataset, NovaDataLoader
from src.training.corpus_processor import CorpusProcessor
from src.training.train_nova import NovaTrainer
from src.config.settings import get_settings


def demo_text_training():
    """Demo: Train on text data (simple tokenization)."""
    print("\n" + "=" * 60)
    print("Demo 1: Text-based Training")
    print("=" * 60)
    
    # Sample training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks learn patterns from data.",
        "Transformers revolutionized natural language processing.",
    ] * 10  # Repeat for more data
    
    # Create dataset
    dataset = NovaDataset(
        data=texts,
        mode="text",
        max_seq_length=128,
        vocab_size=10000,
    )
    
    dataloader = NovaDataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches: {len(dataloader)}")
    
    # Create small model
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_len=128,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = NovaTrainer(
        model=model,
        train_loader=dataloader,
        learning_rate=1e-4,
        mixed_precision=False,  # CPU demo
        device="cpu",
    )
    
    # Train for 2 epochs
    print("\nTraining...")
    history = trainer.train(num_epochs=2)
    
    print(f"\n✓ Training complete!")
    print(f"Final loss: {history['train_history']['loss'][-1]:.4f}")


def demo_ai2ai_training():
    """Demo: Train on AI2AI embeddings from Claude."""
    print("\n" + "=" * 60)
    print("Demo 2: AI2AI Embedding-based Training")
    print("=" * 60)
    
    # Check if API key is available
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("⚠️  ANTHROPIC_API_KEY not found - skipping AI2AI demo")
        print("To run this demo:")
        print("1. Set ANTHROPIC_API_KEY in .env")
        print("2. Run: python examples/training_demo.py")
        return
    
    # Sample corpus
    corpus = [
        "Force equals mass times acceleration (F=ma).",
        "Energy is conserved in isolated systems.",
        "Momentum is the product of mass and velocity.",
        "Newton's laws describe motion and forces.",
        "Gravity causes objects to attract each other.",
    ]
    
    print(f"\nCorpus: {len(corpus)} texts")
    
    # Process corpus with Claude (ONE-TIME)
    print("\nStep 1: Processing corpus with Claude...")
    print("(This is the ONE-TIME step where Claude helps)")
    
    processor = CorpusProcessor(
        embedding_dim=768,
        batch_size=5,
        verbose=True,
    )
    
    # Convert texts → AI2AI embeddings
    messages = processor.process_texts(corpus)
    
    print(f"\n✓ Generated {len(messages)} AI2AI messages")
    
    # Save embeddings (optional)
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_embeddings(messages, output_dir / "physics_embeddings.ai2ai")
    
    # Create dataset from embeddings
    print("\nStep 2: Creating dataset from embeddings...")
    dataset = NovaDataset(
        data=messages,
        mode="ai2ai",
        max_seq_length=512,
        embedding_dim=768,
    )
    
    dataloader = NovaDataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )
    
    # Create NOVA model
    print("\nStep 3: Creating NOVA model...")
    model = Transformer(
        src_vocab_size=50000,  # Not used in embedding mode
        tgt_vocab_size=50000,
        d_model=768,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=2048,
        max_len=512,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = NovaTrainer(
        model=model,
        train_loader=dataloader,
        learning_rate=1e-4,
        mixed_precision=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Train
    print("\nStep 4: Training NOVA...")
    print("(Now Claude is not needed - NOVA learns from embeddings)")
    history = trainer.train(num_epochs=3)
    
    print(f"\n✓ Training complete!")
    print(f"Final loss: {history['train_history']['loss'][-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("NOVA is now trained!")
    print("Knowledge is embedded in NOVA's parameters.")
    print("NOVA can operate 100% independently (no API calls).")
    print("=" * 60)


def demo_knowledge_extraction():
    """Demo: Extract structured knowledge from Claude."""
    print("\n" + "=" * 60)
    print("Demo 3: Structured Knowledge Extraction")
    print("=" * 60)
    
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("⚠️  ANTHROPIC_API_KEY not found - skipping demo")
        return
    
    processor = CorpusProcessor(verbose=True)
    
    # Extract physics concepts
    concepts = ["force", "mass", "acceleration", "energy", "momentum"]
    
    print(f"\nExtracting knowledge for domain: physics")
    print(f"Concepts: {', '.join(concepts)}")
    
    knowledge = processor.extract_domain_knowledge(
        domain="physics",
        concepts=concepts,
        output_file=Path("data/processed/physics_knowledge.ai2ai")
    )
    
    print(f"\n✓ Extracted {len(knowledge.concept_names)} concepts")
    print(f"Confidence: {knowledge.confidence}")
    print(f"Relationships shape: {knowledge.relationships.shape if knowledge.relationships is not None else 'None'}")
    
    # This knowledge can be used for targeted training
    print("\nThis structured knowledge can be used to:")
    print("- Pre-train NOVA on specific domains")
    print("- Fine-tune with targeted concept learning")
    print("- Build domain-specific models")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("NOVA Training Pipeline Demos")
    print("=" * 60)
    
    # Demo 1: Simple text training (no API needed)
    demo_text_training()
    
    # Demo 2: AI2AI embedding training (requires API key)
    # demo_ai2ai_training()  # Uncomment to run
    
    # Demo 3: Knowledge extraction (requires API key)
    # demo_knowledge_extraction()  # Uncomment to run
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your corpus (books, papers, articles)")
    print("2. Process with Claude ONE-TIME (corpus_processor.py)")
    print("3. Train NOVA on embeddings (train_nova.py)")
    print("4. Deploy NOVA independently (no API costs!)")
    print("=" * 60)


if __name__ == "__main__":
    main()
