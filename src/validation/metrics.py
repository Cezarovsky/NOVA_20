"""
Core Validation Metrics for NOVA

Implements fundamental metrics for language model evaluation:
- Perplexity: Measures model confidence/uncertainty
- Accuracy: Token-level and sequence-level accuracy
- Loss tracking: Training and validation loss
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
import math


class Metric:
    """Base class for all metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        """Update metric with new data."""
        raise NotImplementedError
    
    def compute(self) -> float:
        """Compute final metric value."""
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Perplexity(Metric):
    """
    Perplexity metric for language models.
    
    Perplexity = exp(average_cross_entropy_loss)
    
    Lower perplexity = better model (more confident predictions)
    Perplexity of 1 = perfect predictions
    Random baseline perplexity = vocabulary_size
    
    Example:
        >>> perplexity = Perplexity()
        >>> for batch in dataloader:
        ...     outputs = model(batch['input_ids'])
        ...     perplexity.update(outputs, batch['labels'])
        >>> print(f"Perplexity: {perplexity.compute():.2f}")
    """
    
    def __init__(self, ignore_index: int = -100):
        """
        Initialize perplexity metric.
        
        Args:
            ignore_index: Index to ignore in loss calculation (padding)
        """
        super().__init__("perplexity")
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated values."""
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[torch.Tensor] = None
    ):
        """
        Update perplexity with batch.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            loss: Pre-computed loss (optional)
        """
        if loss is None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.ignore_index,
                reduction='sum'
            )
        
        # Count non-ignored tokens
        mask = (labels != self.ignore_index)
        num_tokens = mask.sum().item()
        
        # Accumulate
        if isinstance(loss, torch.Tensor):
            self.total_loss += loss.item()
        else:
            self.total_loss += loss
        
        self.total_tokens += num_tokens
    
    def compute(self) -> float:
        """Compute perplexity."""
        if self.total_tokens == 0:
            return float('inf')
        
        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_from_loss(self, loss: float) -> float:
        """Compute perplexity directly from loss."""
        return math.exp(loss)


class Accuracy(Metric):
    """
    Token-level accuracy metric.
    
    Measures percentage of correctly predicted tokens.
    """
    
    def __init__(self, ignore_index: int = -100, top_k: int = 1):
        """
        Initialize accuracy metric.
        
        Args:
            ignore_index: Index to ignore (padding)
            top_k: Consider top-k predictions (default: 1 for exact match)
        """
        super().__init__("accuracy")
        self.ignore_index = ignore_index
        self.top_k = top_k
        self.reset()
    
    def reset(self):
        """Reset counters."""
        self.correct = 0
        self.total = 0
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update accuracy with batch.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
        """
        # Get predictions
        if self.top_k == 1:
            predictions = logits.argmax(dim=-1)
            matches = (predictions == labels)
        else:
            # Top-k accuracy
            _, top_k_preds = logits.topk(self.top_k, dim=-1)
            matches = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1)
        
        # Mask padding tokens
        mask = (labels != self.ignore_index)
        matches = matches & mask
        
        # Accumulate
        self.correct += matches.sum().item()
        self.total += mask.sum().item()
    
    def compute(self) -> float:
        """Compute accuracy percentage."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total


class TokenAccuracy(Metric):
    """
    Detailed token-level accuracy with per-position tracking.
    
    Tracks accuracy at each position in sequence.
    Useful for analyzing where model struggles.
    """
    
    def __init__(self, max_seq_len: int = 512, ignore_index: int = -100):
        """
        Initialize token accuracy.
        
        Args:
            max_seq_len: Maximum sequence length to track
            ignore_index: Index to ignore
        """
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        super().__init__("token_accuracy")
        self.reset()
    
    def reset(self):
        """Reset position-wise counters."""
        self.position_correct = np.zeros(self.max_seq_len)
        self.position_total = np.zeros(self.max_seq_len)
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update position-wise accuracy.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
        """
        predictions = logits.argmax(dim=-1)
        matches = (predictions == labels)
        mask = (labels != self.ignore_index)
        
        # Per-position accuracy
        seq_len = min(labels.size(1), self.max_seq_len)
        
        for pos in range(seq_len):
            pos_matches = matches[:, pos] & mask[:, pos]
            pos_valid = mask[:, pos]
            
            self.position_correct[pos] += pos_matches.sum().item()
            self.position_total[pos] += pos_valid.sum().item()
    
    def compute(self) -> float:
        """Compute overall accuracy."""
        valid = self.position_total > 0
        if not valid.any():
            return 0.0
        
        total_correct = self.position_correct[valid].sum()
        total = self.position_total[valid].sum()
        
        return 100.0 * total_correct / total if total > 0 else 0.0
    
    def compute_per_position(self) -> np.ndarray:
        """Compute accuracy at each position."""
        valid = self.position_total > 0
        accuracy = np.zeros_like(self.position_correct)
        accuracy[valid] = 100.0 * self.position_correct[valid] / self.position_total[valid]
        return accuracy


class BLEU:
    """
    BLEU (Bilingual Evaluation Understudy) metric.
    
    Measures n-gram overlap between generated and reference texts.
    Commonly used for machine translation evaluation.
    
    BLEU score ranges from 0 to 100 (higher is better).
    """
    
    def __init__(self, n_grams: int = 4, smooth: bool = True):
        """
        Initialize BLEU metric.
        
        Args:
            n_grams: Maximum n-gram size (typically 4)
            smooth: Apply smoothing for zero counts
        """
        self.n_grams = n_grams
        self.smooth = smooth
    
    def compute(
        self,
        hypotheses: List[List[str]],
        references: List[List[List[str]]],
    ) -> float:
        """
        Compute BLEU score.
        
        Args:
            hypotheses: List of generated token sequences
            references: List of reference token sequences (can have multiple refs per hypothesis)
            
        Returns:
            BLEU score (0-100)
        """
        from collections import Counter
        
        # Compute n-gram precisions
        precisions = []
        
        for n in range(1, self.n_grams + 1):
            matches = 0
            total = 0
            
            for hyp, refs in zip(hypotheses, references):
                # Get n-grams from hypothesis
                hyp_ngrams = self._get_ngrams(hyp, n)
                
                # Get max counts from all references
                ref_ngrams_list = [self._get_ngrams(ref, n) for ref in refs]
                max_ref_counts = {}
                
                for ref_ngrams in ref_ngrams_list:
                    for ngram, count in ref_ngrams.items():
                        max_ref_counts[ngram] = max(
                            max_ref_counts.get(ngram, 0),
                            count
                        )
                
                # Count matches (clipped)
                for ngram, count in hyp_ngrams.items():
                    matches += min(count, max_ref_counts.get(ngram, 0))
                
                total += len(hyp) - n + 1
            
            # Smoothing for zero counts
            if total == 0:
                precision = 0.0
            elif matches == 0 and self.smooth:
                precision = 1.0 / (2 ** n * total)
            else:
                precision = matches / total if total > 0 else 0.0
            
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            geo_mean = 0.0
        
        # Brevity penalty
        hyp_len = sum(len(hyp) for hyp in hypotheses)
        ref_len = sum(
            min(len(ref) for ref in refs)
            for refs in references
        )
        
        if hyp_len > ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        bleu = 100.0 * bp * geo_mean
        return bleu
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Extract n-grams from token sequence."""
        from collections import Counter
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        
        return Counter(ngrams)


class ROUGE:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
    
    Measures recall of n-grams between generated and reference texts.
    Commonly used for summarization evaluation.
    
    Variants: ROUGE-N (n-gram), ROUGE-L (longest common subsequence)
    """
    
    def __init__(self, rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_types: Types of ROUGE to compute
        """
        self.rouge_types = rouge_types
    
    def compute(
        self,
        hypotheses: List[List[str]],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            hypotheses: Generated token sequences
            references: Reference token sequences
            
        Returns:
            Dictionary with ROUGE scores
        """
        scores = {}
        
        for rouge_type in self.rouge_types:
            if rouge_type == "rouge1":
                scores[rouge_type] = self._rouge_n(hypotheses, references, n=1)
            elif rouge_type == "rouge2":
                scores[rouge_type] = self._rouge_n(hypotheses, references, n=2)
            elif rouge_type == "rougeL":
                scores[rouge_type] = self._rouge_l(hypotheses, references)
        
        return scores
    
    def _rouge_n(
        self,
        hypotheses: List[List[str]],
        references: List[List[str]],
        n: int
    ) -> float:
        """Compute ROUGE-N (n-gram overlap recall)."""
        total_recall = 0.0
        
        for hyp, ref in zip(hypotheses, references):
            hyp_ngrams = set(self._get_ngrams(hyp, n))
            ref_ngrams = set(self._get_ngrams(ref, n))
            
            if len(ref_ngrams) == 0:
                recall = 0.0
            else:
                matches = len(hyp_ngrams & ref_ngrams)
                recall = matches / len(ref_ngrams)
            
            total_recall += recall
        
        return 100.0 * total_recall / len(hypotheses) if hypotheses else 0.0
    
    def _rouge_l(
        self,
        hypotheses: List[List[str]],
        references: List[List[str]]
    ) -> float:
        """Compute ROUGE-L (longest common subsequence)."""
        total_f1 = 0.0
        
        for hyp, ref in zip(hypotheses, references):
            lcs_len = self._lcs_length(hyp, ref)
            
            if len(ref) == 0 or len(hyp) == 0:
                f1 = 0.0
            else:
                recall = lcs_len / len(ref)
                precision = lcs_len / len(hyp)
                
                if recall + precision == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * recall * precision / (recall + precision)
            
            total_f1 += f1
        
        return 100.0 * total_f1 / len(hypotheses) if hypotheses else 0.0
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class EmbeddingSimilarity(Metric):
    """
    Embedding similarity metrics.
    
    Measures similarity between predicted and target embeddings.
    Useful for AI2AI training evaluation.
    """
    
    def __init__(self, metric_type: str = "cosine"):
        """
        Initialize embedding similarity.
        
        Args:
            metric_type: "cosine", "l2", or "dot"
        """
        super().__init__(f"embedding_{metric_type}")
        self.metric_type = metric_type
        self.reset()
    
    def reset(self):
        """Reset accumulator."""
        self.total_similarity = 0.0
        self.count = 0
    
    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Update similarity with batch.
        
        Args:
            predicted: Predicted embeddings [batch, seq_len, dim]
            target: Target embeddings [batch, seq_len, dim]
            mask: Optional mask [batch, seq_len]
        """
        if self.metric_type == "cosine":
            # Cosine similarity
            similarity = F.cosine_similarity(predicted, target, dim=-1)
        
        elif self.metric_type == "l2":
            # Negative L2 distance (higher is better)
            similarity = -torch.norm(predicted - target, p=2, dim=-1)
        
        elif self.metric_type == "dot":
            # Dot product
            similarity = (predicted * target).sum(dim=-1)
        
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
        
        # Apply mask if provided
        if mask is not None:
            similarity = similarity * mask
            count = mask.sum().item()
        else:
            count = similarity.numel()
        
        self.total_similarity += similarity.sum().item()
        self.count += count
    
    def compute(self) -> float:
        """Compute average similarity."""
        if self.count == 0:
            return 0.0
        return self.total_similarity / self.count


class MetricsTracker:
    """
    Track multiple metrics over training/validation.
    
    Provides unified interface for metric collection and logging.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.history = defaultdict(list)
    
    def add_metric(self, metric: Metric):
        """Add metric to tracker."""
        self.metrics[metric.name] = metric
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, metric_name: str, *args, **kwargs):
        """Update specific metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].update(*args, **kwargs)
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
            self.history[name].append(results[name])
        return results
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get history for specific metric."""
        return self.history[metric_name]
    
    def summary(self) -> str:
        """Get summary string of current metrics."""
        results = self.compute_all()
        lines = ["Metrics Summary:", "=" * 50]
        for name, value in results.items():
            lines.append(f"{name:20s}: {value:8.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
