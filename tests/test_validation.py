"""
Tests for NOVA Validation & Metrics

Comprehensive test suite for all validation components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.metrics import (
    Perplexity,
    Accuracy,
    TokenAccuracy,
    BLEU,
    ROUGE,
    EmbeddingSimilarity,
    MetricsTracker,
)


class TestPerplexity:
    """Test perplexity metric."""
    
    def test_perfect_predictions(self):
        """Test perplexity with perfect predictions."""
        perplexity = Perplexity()
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        # Create perfect predictions
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Set logits to very high for correct tokens
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 100.0
        
        perplexity.update(logits, labels)
        ppl = perplexity.compute()
        
        # Perfect predictions should have perplexity close to 1
        assert ppl < 1.1, f"Expected perplexity < 1.1, got {ppl}"
    
    def test_random_predictions(self):
        """Test perplexity with random predictions."""
        perplexity = Perplexity()
        
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        perplexity.update(logits, labels)
        ppl = perplexity.compute()
        
        # Random predictions should have high perplexity
        assert ppl > 10, f"Expected perplexity > 10, got {ppl}"
    
    def test_ignore_padding(self):
        """Test that padding tokens are ignored."""
        perplexity = Perplexity(ignore_index=-100)
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Mark half as padding
        labels[:, 5:] = -100
        
        perplexity.update(logits, labels)
        ppl = perplexity.compute()
        
        assert ppl < float('inf'), "Perplexity should be finite"
        assert ppl > 0, "Perplexity should be positive"
    
    def test_reset(self):
        """Test metric reset."""
        perplexity = Perplexity()
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        perplexity.update(logits, labels)
        ppl1 = perplexity.compute()
        
        perplexity.reset()
        perplexity.update(logits, labels)
        ppl2 = perplexity.compute()
        
        assert ppl1 == ppl2, "Reset should give same result"


class TestAccuracy:
    """Test accuracy metric."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        accuracy = Accuracy()
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Set correct predictions
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 100.0
        
        accuracy.update(logits, labels)
        acc = accuracy.compute()
        
        assert acc == 100.0, f"Expected 100% accuracy, got {acc}%"
    
    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        accuracy = Accuracy()
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Make sure wrong token is predicted
        logits[:, :, 0] = -100.0
        logits[:, :, 1] = 100.0
        
        accuracy.update(logits, labels)
        acc = accuracy.compute()
        
        assert acc == 0.0, f"Expected 0% accuracy, got {acc}%"
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy."""
        accuracy = Accuracy(top_k=5)
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Ensure correct label is in top-5
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 50.0  # High but not highest
        
        accuracy.update(logits, labels)
        acc = accuracy.compute()
        
        assert acc == 100.0, f"Expected 100% top-5 accuracy, got {acc}%"


class TestBLEU:
    """Test BLEU metric."""
    
    def test_perfect_match(self):
        """Test BLEU with perfect match."""
        bleu = BLEU(n_grams=4)
        
        hypotheses = [["the", "cat", "sat", "on", "the", "mat"]]
        references = [[["the", "cat", "sat", "on", "the", "mat"]]]
        
        score = bleu.compute(hypotheses, references)
        
        assert score == 100.0, f"Expected BLEU=100, got {score}"
    
    def test_no_match(self):
        """Test BLEU with no n-gram overlap."""
        bleu = BLEU(n_grams=4, smooth=False)
        
        hypotheses = [["a", "b", "c", "d"]]
        references = [[["e", "f", "g", "h"]]]
        
        score = bleu.compute(hypotheses, references)
        
        assert score == 0.0, f"Expected BLEU=0, got {score}"
    
    def test_partial_match(self):
        """Test BLEU with partial match."""
        bleu = BLEU(n_grams=2)
        
        hypotheses = [["the", "cat", "sat"]]
        references = [[["the", "cat", "slept"]]]
        
        score = bleu.compute(hypotheses, references)
        
        # Should have partial overlap (unigrams and one bigram)
        assert 0 < score < 100, f"Expected 0 < BLEU < 100, got {score}"


class TestROUGE:
    """Test ROUGE metric."""
    
    def test_perfect_match(self):
        """Test ROUGE with perfect match."""
        rouge = ROUGE(rouge_types=["rouge1"])
        
        hypotheses = [["the", "cat", "sat"]]
        references = [["the", "cat", "sat"]]
        
        scores = rouge.compute(hypotheses, references)
        
        assert scores['rouge1'] == 100.0, f"Expected ROUGE-1=100, got {scores['rouge1']}"
    
    def test_rouge_l(self):
        """Test ROUGE-L (longest common subsequence)."""
        rouge = ROUGE(rouge_types=["rougeL"])
        
        hypotheses = [["a", "b", "c", "d", "e"]]
        references = [["a", "b", "x", "c", "d"]]
        
        scores = rouge.compute(hypotheses, references)
        
        # LCS is "a b c d" (length 4)
        assert scores['rougeL'] > 0, f"Expected ROUGE-L > 0, got {scores['rougeL']}"


class TestEmbeddingSimilarity:
    """Test embedding similarity metrics."""
    
    def test_cosine_identical(self):
        """Test cosine similarity with identical embeddings."""
        metric = EmbeddingSimilarity(metric_type="cosine")
        
        embeddings = torch.randn(2, 10, 768)
        
        metric.update(embeddings, embeddings)
        similarity = metric.compute()
        
        assert abs(similarity - 1.0) < 0.01, f"Expected similarity ≈ 1.0, got {similarity}"
    
    def test_l2_identical(self):
        """Test L2 distance with identical embeddings."""
        metric = EmbeddingSimilarity(metric_type="l2")
        
        embeddings = torch.randn(2, 10, 768)
        
        metric.update(embeddings, embeddings)
        distance = metric.compute()
        
        assert abs(distance) < 0.01, f"Expected distance ≈ 0.0, got {distance}"
    
    def test_with_mask(self):
        """Test similarity with attention mask."""
        metric = EmbeddingSimilarity(metric_type="cosine")
        
        predicted = torch.randn(2, 10, 768)
        target = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask second half
        
        metric.update(predicted, target, mask)
        similarity = metric.compute()
        
        assert similarity != 0.0, "Similarity should be computed for masked regions"


class TestMetricsTracker:
    """Test metrics tracker."""
    
    def test_add_metrics(self):
        """Test adding multiple metrics."""
        tracker = MetricsTracker()
        
        tracker.add_metric(Perplexity())
        tracker.add_metric(Accuracy())
        
        assert len(tracker.metrics) == 2
        assert 'perplexity' in tracker.metrics
        assert 'accuracy' in tracker.metrics
    
    def test_update_and_compute(self):
        """Test updating and computing metrics."""
        tracker = MetricsTracker()
        tracker.add_metric(Perplexity())
        tracker.add_metric(Accuracy())
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        tracker.update('perplexity', logits, labels)
        tracker.update('accuracy', logits, labels)
        
        results = tracker.compute_all()
        
        assert 'perplexity' in results
        assert 'accuracy' in results
        assert results['perplexity'] > 0
        assert 0 <= results['accuracy'] <= 100
    
    def test_history_tracking(self):
        """Test history tracking."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        # Update 3 times
        for _ in range(3):
            tracker.update('accuracy', logits, labels)
            tracker.compute_all()
        
        history = tracker.get_history('accuracy')
        assert len(history) == 3


class TestIntegration:
    """Integration tests for validation system."""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        # Create tracker with multiple metrics
        tracker = MetricsTracker()
        tracker.add_metric(Perplexity())
        tracker.add_metric(Accuracy())
        tracker.add_metric(TokenAccuracy(max_seq_len=32))
        
        # Simulate validation loop
        num_batches = 5
        for _ in range(num_batches):
            logits = torch.randn(2, 32, 1000)
            labels = torch.randint(0, 1000, (2, 32))
            
            tracker.update('perplexity', logits, labels)
            tracker.update('accuracy', logits, labels)
            tracker.update('token_accuracy', logits, labels)
        
        # Compute all
        results = tracker.compute_all()
        
        assert len(results) == 3
        assert all(isinstance(v, float) for v in results.values())
        
        # Summary should be printable
        summary = tracker.summary()
        assert isinstance(summary, str)
        assert 'perplexity' in summary.lower()
    
    def test_save_and_load_results(self):
        """Test saving validation results."""
        results = {
            'perplexity': 45.2,
            'accuracy': 78.5,
            'loss': 3.82,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            
            # Save
            with open(filepath, 'w') as f:
                json.dump(results, f)
            
            # Load
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            assert loaded == results


@pytest.mark.integration
def test_validation_demo_runs():
    """Test that validation demo runs without errors."""
    from examples.validation_demo import (
        demo_core_metrics,
        demo_generation_metrics,
        demo_embedding_metrics,
        demo_metrics_tracker,
    )
    
    # Run each demo (except TensorBoard which requires filesystem)
    demo_core_metrics()
    demo_generation_metrics()
    demo_embedding_metrics()
    demo_metrics_tracker()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
