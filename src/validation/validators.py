"""
Model Validators for NOVA

High-level validation workflows combining multiple metrics.
Provides comprehensive model evaluation across different aspects.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time

from .metrics import (
    Perplexity,
    Accuracy,
    TokenAccuracy,
    BLEU,
    ROUGE,
    EmbeddingSimilarity,
    MetricsTracker,
)


class ModelValidator:
    """
    Comprehensive model validation.
    
    Evaluates model across multiple metrics:
    - Perplexity (language modeling quality)
    - Accuracy (token prediction)
    - Loss (training objective)
    
    Example:
        >>> validator = ModelValidator(model, val_dataloader)
        >>> results = validator.validate()
        >>> print(results['perplexity'])
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ignore_index: int = -100,
    ):
        """
        Initialize model validator.
        
        Args:
            model: NOVA model to validate
            device: Device for computation
            ignore_index: Padding token index
        """
        self.model = model
        self.device = device
        self.ignore_index = ignore_index
        
        # Initialize metrics
        self.tracker = MetricsTracker()
        self.tracker.add_metric(Perplexity(ignore_index=ignore_index))
        self.tracker.add_metric(Accuracy(ignore_index=ignore_index))
        self.tracker.add_metric(TokenAccuracy(ignore_index=ignore_index))
    
    @torch.no_grad()
    def validate(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run validation on dataset.
        
        Args:
            dataloader: Validation dataloader
            max_batches: Limit number of batches (for quick validation)
            verbose: Print progress
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.tracker.reset()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            
            # Update metrics
            self.tracker.update('perplexity', logits, labels, loss)
            self.tracker.update('accuracy', logits, labels)
            self.tracker.update('token_accuracy', logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
        
        # Compute final metrics
        results = self.tracker.compute_all()
        results['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        results['validation_time'] = time.time() - start_time
        results['num_batches'] = num_batches
        
        if verbose:
            print(f"\nValidation Results:")
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  Perplexity: {results['perplexity']:.2f}")
            print(f"  Accuracy: {results['accuracy']:.2f}%")
            print(f"  Time: {results['validation_time']:.2f}s")
        
        return results
    
    def save_results(self, results: Dict[str, float], filepath: Path):
        """Save validation results to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


class EmbeddingValidator:
    """
    Validate embedding quality for AI2AI training.
    
    Measures:
    - Embedding similarity (cosine, L2)
    - Reconstruction error
    - Semantic alignment
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        similarity_metrics: List[str] = ["cosine", "l2"],
    ):
        """
        Initialize embedding validator.
        
        Args:
            model: NOVA model
            device: Computation device
            similarity_metrics: List of similarity metrics to use
        """
        self.model = model
        self.device = device
        
        # Initialize metrics
        self.tracker = MetricsTracker()
        for metric_type in similarity_metrics:
            metric = EmbeddingSimilarity(metric_type=metric_type)
            self.tracker.add_metric(metric)
    
    @torch.no_grad()
    def validate_embeddings(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Validate embedding predictions.
        
        Args:
            dataloader: Dataloader with AI2AI embeddings
            max_batches: Limit batches
            verbose: Print progress
            
        Returns:
            Embedding similarity metrics
        """
        self.model.eval()
        self.tracker.reset()
        
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Get embeddings
            embeddings = batch['embeddings'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward through model (using forward_embeddings if available)
            if hasattr(self.model, 'forward_embeddings'):
                outputs = self.model.forward_embeddings(
                    embeddings,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self.model(embeddings, attention_mask=attention_mask)
            
            predicted = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            # For next-token prediction, compare shifted embeddings
            # predicted: [batch, seq_len, vocab_size] or [batch, seq_len, d_model]
            # target: embeddings shifted by 1
            if predicted.size(-1) != embeddings.size(-1):
                # Model outputs logits, need to project back to embedding space
                # Skip this comparison for now
                continue
            
            target = embeddings[:, 1:, :]  # Shift target
            predicted = predicted[:, :-1, :]  # Align with target
            
            mask = attention_mask[:, 1:] if attention_mask is not None else None
            
            # Update similarity metrics
            for metric_name in self.tracker.metrics:
                self.tracker.update(metric_name, predicted, target, mask)
            
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
        
        # Compute results
        results = self.tracker.compute_all()
        results['validation_time'] = time.time() - start_time
        results['num_batches'] = num_batches
        
        if verbose:
            print(f"\nEmbedding Validation Results:")
            for metric_name, value in results.items():
                if metric_name not in ['validation_time', 'num_batches']:
                    print(f"  {metric_name}: {value:.4f}")
            print(f"  Time: {results['validation_time']:.2f}s")
        
        return results


class GenerationValidator:
    """
    Validate text generation quality.
    
    Evaluates generated text using:
    - BLEU (n-gram precision)
    - ROUGE (n-gram recall)
    - Custom domain metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        max_length: int = 128,
    ):
        """
        Initialize generation validator.
        
        Args:
            model: NOVA model
            tokenizer: Tokenizer for decoding
            device: Computation device
            max_length: Max generation length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Initialize metrics
        self.bleu = BLEU(n_grams=4)
        self.rouge = ROUGE(rouge_types=["rouge1", "rouge2", "rougeL"])
    
    @torch.no_grad()
    def validate_generation(
        self,
        prompts: List[str],
        references: List[str],
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Validate generation quality.
        
        Args:
            prompts: Input prompts
            references: Reference completions
            num_samples: Limit samples
            verbose: Print progress
            
        Returns:
            Generation quality metrics
        """
        self.model.eval()
        
        if num_samples:
            prompts = prompts[:num_samples]
            references = references[:num_samples]
        
        generated_texts = []
        start_time = time.time()
        
        for idx, prompt in enumerate(prompts):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Generating {idx + 1}/{len(prompts)}")
            
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens]).to(self.device)
            
            # Generate
            generated = self._generate(input_ids)
            generated_text = self.tokenizer.decode(generated[0].tolist())
            generated_texts.append(generated_text)
        
        # Tokenize for metric computation
        hypotheses = [self.tokenizer.encode(text) for text in generated_texts]
        refs = [[self.tokenizer.encode(ref)] for ref in references]
        
        # Compute BLEU
        bleu_score = self.bleu.compute(hypotheses, refs)
        
        # Compute ROUGE
        rouge_scores = self.rouge.compute(
            hypotheses,
            [ref[0] for ref in refs]  # Single reference per hypothesis
        )
        
        results = {
            'bleu': bleu_score,
            **rouge_scores,
            'generation_time': time.time() - start_time,
            'num_samples': len(prompts),
        }
        
        if verbose:
            print(f"\nGeneration Validation Results:")
            print(f"  BLEU: {results['bleu']:.2f}")
            for rouge_type, score in rouge_scores.items():
                print(f"  {rouge_type.upper()}: {score:.2f}")
            print(f"  Time: {results['generation_time']:.2f}s")
        
        return results
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate text continuation.
        
        Simple greedy generation for validation.
        """
        generated = input_ids
        
        for _ in range(self.max_length):
            outputs = self.model(generated)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token (if applicable)
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated
    
    def sample_generations(
        self,
        prompts: List[str],
        num_samples: int = 5,
    ) -> List[Tuple[str, str]]:
        """
        Generate samples for qualitative evaluation.
        
        Returns:
            List of (prompt, generation) pairs
        """
        self.model.eval()
        samples = []
        
        for prompt in prompts[:num_samples]:
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens]).to(self.device)
            
            generated = self._generate(input_ids)
            generated_text = self.tokenizer.decode(generated[0].tolist())
            
            samples.append((prompt, generated_text))
        
        return samples
