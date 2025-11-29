"""
Validation & Metrics Demo

Demonstrates comprehensive validation and metrics tracking for NOVA.

Usage:
    python examples/validation_demo.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import (
    Perplexity,
    Accuracy,
    TokenAccuracy,
    BLEU,
    ROUGE,
    EmbeddingSimilarity,
    MetricsTracker,
    ModelValidator,
    EmbeddingValidator,
    GenerationValidator,
    LanguageModelingBenchmark,
    DomainBenchmark,
    BenchmarkSuite,
)
from src.validation.tensorboard_logger import TensorBoardLogger, MetricsDashboard
from src.ml.transformer import Transformer
from src.ml.embeddings import TokenEmbedding


def demo_core_metrics():
    """Demonstrate core metrics: perplexity, accuracy."""
    print("\n" + "="*60)
    print("DEMO 1: Core Metrics (Perplexity, Accuracy)")
    print("="*60)
    
    # Simulate model outputs
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, -1] = -100  # Ignore last token (padding)
    
    # Perplexity
    perplexity = Perplexity()
    perplexity.update(logits, labels)
    ppl = perplexity.compute()
    
    print(f"\nPerplexity: {ppl:.2f}")
    print(f"  (Lower is better, random baseline: {vocab_size:.0f})")
    
    # Accuracy
    accuracy = Accuracy()
    accuracy.update(logits, labels)
    acc = accuracy.compute()
    
    print(f"\nToken Accuracy: {acc:.2f}%")
    print(f"  (Random baseline: {100.0/vocab_size:.2f}%)")
    
    # Token-level accuracy with position tracking
    token_acc = TokenAccuracy(max_seq_len=seq_len)
    token_acc.update(logits, labels)
    
    print(f"\nPosition-wise Accuracy:")
    pos_acc = token_acc.compute_per_position()
    for pos, acc_val in enumerate(pos_acc):
        if pos < seq_len - 1:  # Skip padding
            print(f"  Position {pos}: {acc_val:.2f}%")


def demo_generation_metrics():
    """Demonstrate BLEU and ROUGE metrics."""
    print("\n" + "="*60)
    print("DEMO 2: Generation Metrics (BLEU, ROUGE)")
    print("="*60)
    
    # Sample data
    hypotheses = [
        "the cat sat on the mat".split(),
        "a quick brown fox jumps over lazy dog".split(),
    ]
    
    references = [
        ["the cat is on the mat".split()],  # Single reference (as list of tokens)
        ["a fast brown fox jumped over the lazy dog".split()],
    ]
    
    # BLEU
    bleu = BLEU(n_grams=4)
    bleu_score = bleu.compute(hypotheses, references)
    
    print(f"\nBLEU Score: {bleu_score:.2f}")
    print("  (0-100 scale, higher is better)")
    print("  Measures n-gram precision overlap")
    
    # ROUGE
    rouge = ROUGE(rouge_types=["rouge1", "rouge2", "rougeL"])
    rouge_scores = rouge.compute(
        hypotheses,
        [ref[0] for ref in references]  # Flatten references
    )
    
    print(f"\nROUGE Scores:")
    for metric_name, score in rouge_scores.items():
        print(f"  {metric_name.upper()}: {score:.2f}")
    print("  (Measures n-gram recall overlap)")


def demo_embedding_metrics():
    """Demonstrate embedding similarity metrics."""
    print("\n" + "="*60)
    print("DEMO 3: Embedding Similarity Metrics")
    print("="*60)
    
    # Simulate embeddings
    batch_size, seq_len, dim = 2, 10, 768
    
    predicted = torch.randn(batch_size, seq_len, dim)
    target = predicted + torch.randn(batch_size, seq_len, dim) * 0.1  # Add noise
    
    # Cosine similarity
    cos_sim = EmbeddingSimilarity(metric_type="cosine")
    cos_sim.update(predicted, target)
    cos_score = cos_sim.compute()
    
    print(f"\nCosine Similarity: {cos_score:.4f}")
    print("  (Range: -1 to 1, higher is better)")
    
    # L2 distance
    l2_sim = EmbeddingSimilarity(metric_type="l2")
    l2_sim.update(predicted, target)
    l2_score = l2_sim.compute()
    
    print(f"\nL2 Distance: {l2_score:.4f}")
    print("  (Negative distance, closer to 0 is better)")
    
    # Dot product
    dot_sim = EmbeddingSimilarity(metric_type="dot")
    dot_sim.update(predicted, target)
    dot_score = dot_sim.compute()
    
    print(f"\nDot Product: {dot_score:.4f}")
    print("  (Higher is better)")


def demo_metrics_tracker():
    """Demonstrate MetricsTracker for unified tracking."""
    print("\n" + "="*60)
    print("DEMO 4: Metrics Tracker")
    print("="*60)
    
    # Initialize tracker
    tracker = MetricsTracker()
    tracker.add_metric(Perplexity())
    tracker.add_metric(Accuracy())
    tracker.add_metric(EmbeddingSimilarity(metric_type="cosine"))
    
    print("\nRegistered metrics:")
    for name in tracker.metrics.keys():
        print(f"  - {name}")
    
    # Simulate training loop
    print("\nSimulating 3 training steps...")
    
    for step in range(3):
        # Generate fake data
        batch_size, seq_len, vocab_size, dim = 2, 10, 1000, 768
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        embeddings_pred = torch.randn(batch_size, seq_len, dim)
        embeddings_target = embeddings_pred + torch.randn(batch_size, seq_len, dim) * 0.1
        
        # Update metrics
        tracker.update('perplexity', logits, labels)
        tracker.update('accuracy', logits, labels)
        tracker.update('embedding_cosine', embeddings_pred, embeddings_target)
        
        # Compute and print
        results = tracker.compute_all()
        print(f"\nStep {step + 1}:")
        for name, value in results.items():
            print(f"  {name}: {value:.4f}")
        
        # Reset for next step
        tracker.reset()
    
    # Print summary
    print("\n" + tracker.summary())


def demo_tensorboard_logging():
    """Demonstrate TensorBoard logging."""
    print("\n" + "="*60)
    print("DEMO 5: TensorBoard Logging")
    print("="*60)
    
    # Initialize logger
    log_dir = Path(__file__).parent.parent / "runs"
    
    with TensorBoardLogger(
        log_dir=str(log_dir),
        experiment_name="validation_demo",
        comment="metrics_showcase"
    ) as logger:
        
        print("\nLogging metrics to TensorBoard...")
        
        # Simulate training
        for step in range(100):
            # Fake metrics
            loss = 2.0 * (0.95 ** step) + 0.1  # Decreasing loss
            perplexity = torch.exp(torch.tensor(loss)).item()
            accuracy = min(95.0, 50.0 + step * 0.5)  # Increasing accuracy
            
            # Log scalars
            logger.log_scalar("loss", loss, step, group="train")
            logger.log_scalar("perplexity", perplexity, step, group="train")
            logger.log_scalar("accuracy", accuracy, step, group="train")
            
            # Log learning rate
            lr = 1e-4 * (0.99 ** (step // 10))
            logger.log_learning_rate(lr, step)
            
            # Validation every 10 steps
            if step % 10 == 0:
                val_loss = loss * 1.1  # Slightly higher than train
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                val_acc = accuracy * 0.95
                
                logger.log_metrics({
                    'loss': val_loss,
                    'perplexity': val_ppl,
                    'accuracy': val_acc,
                }, step, prefix="validation")
                
                print(f"  Step {step}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
        
        # Log model weights histogram (simulate)
        print("\nLogging weight histograms...")
        weights = torch.randn(1000, 768)
        logger.log_histogram("weights/layer_0", weights, step=100)
        
        # Log embeddings
        print("Logging embeddings for projector...")
        embeddings = torch.randn(100, 768)
        metadata = [f"token_{i}" for i in range(100)]
        logger.log_embeddings(embeddings, metadata=metadata, tag="token_embeddings")
        
        print(f"\n✓ Logs saved to: {logger.log_path}")
        print(f"  View with: tensorboard --logdir={log_dir}")


def demo_model_validator():
    """Demonstrate ModelValidator with a simple model."""
    print("\n" + "="*60)
    print("DEMO 6: Model Validator")
    print("="*60)
    
    # Create small NOVA model
    print("\nInitializing NOVA model...")
    
    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 2
    
    embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=0,  # Decoder-only
        num_decoder_layers=num_layers,
        dim_feedforward=512,
        dropout=0.1,
    )
    
    class SimpleNOVA(nn.Module):
        def __init__(self, embedding, transformer, vocab_size):
            super().__init__()
            self.embedding = embedding
            self.transformer = transformer
            self.output_proj = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = self.transformer(tgt=x, tgt_key_padding_mask=None)
            logits = self.output_proj(x)
            return logits
    
    model = SimpleNOVA(embedding, transformer, vocab_size)
    
    # Create fake dataloader
    class FakeDataset:
        def __len__(self):
            return 20
        
        def __getitem__(self, idx):
            seq_len = 32
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            labels = torch.randint(0, vocab_size, (seq_len,))
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones(seq_len),
            }
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(FakeDataset(), batch_size=4, shuffle=False)
    
    # Initialize validator
    validator = ModelValidator(model, device="cpu")
    
    # Run validation
    print("\nRunning validation...")
    results = validator.validate(dataloader, max_batches=5, verbose=True)
    
    print(f"\nValidation complete!")
    print(f"  Results: {results}")


def main():
    """Run all validation demos."""
    print("\n" + "="*70)
    print(" "*15 + "NOVA VALIDATION & METRICS DEMO")
    print("="*70)
    
    # Run demos
    demo_core_metrics()
    demo_generation_metrics()
    demo_embedding_metrics()
    demo_metrics_tracker()
    demo_tensorboard_logging()
    demo_model_validator()
    
    print("\n" + "="*70)
    print("All validation demos complete! ✓")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Core metrics: Perplexity & Accuracy for language modeling")
    print("  2. Generation metrics: BLEU & ROUGE for text quality")
    print("  3. Embedding metrics: Cosine similarity for AI2AI training")
    print("  4. MetricsTracker: Unified interface for metric collection")
    print("  5. TensorBoard: Real-time visualization during training")
    print("  6. ModelValidator: Comprehensive model evaluation")
    print("\nNext: Use these metrics in your training pipeline!")


if __name__ == "__main__":
    main()
