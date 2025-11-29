"""
Benchmarking Suite for NOVA

Domain-specific benchmarks and evaluation protocols:
- Language modeling benchmarks
- Domain-specific tasks (physics, math, code)
- Standard datasets (WikiText, Penn Treebank)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from .metrics import Perplexity, Accuracy
from .validators import ModelValidator


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    
    benchmark_name: str
    dataset_name: str
    num_samples: int
    perplexity: float
    accuracy: float
    loss: float
    time_elapsed: float
    tokens_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def summary(self) -> str:
        """Get formatted summary."""
        lines = [
            f"\n{'='*60}",
            f"Benchmark: {self.benchmark_name}",
            f"Dataset: {self.dataset_name}",
            f"{'='*60}",
            f"Samples: {self.num_samples}",
            f"Perplexity: {self.perplexity:.2f}",
            f"Accuracy: {self.accuracy:.2f}%",
            f"Loss: {self.loss:.4f}",
            f"Time: {self.time_elapsed:.2f}s",
            f"Throughput: {self.tokens_per_second:.0f} tokens/s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class LanguageModelingBenchmark:
    """
    Standard language modeling benchmarks.
    
    Evaluates model on standard datasets:
    - WikiText-2, WikiText-103
    - Penn Treebank
    - Custom corpora
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ignore_index: int = -100,
    ):
        """
        Initialize benchmark.
        
        Args:
            model: NOVA model
            device: Computation device
            ignore_index: Padding index
        """
        self.model = model
        self.device = device
        self.ignore_index = ignore_index
        
        self.validator = ModelValidator(
            model=model,
            device=device,
            ignore_index=ignore_index,
        )
    
    def run_benchmark(
        self,
        dataloader,
        dataset_name: str = "custom",
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run language modeling benchmark.
        
        Args:
            dataloader: Test dataloader
            dataset_name: Name of dataset
            max_batches: Limit batches
            verbose: Print progress
            
        Returns:
            BenchmarkResult with metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Language Modeling Benchmark: {dataset_name}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run validation
        results = self.validator.validate(
            dataloader,
            max_batches=max_batches,
            verbose=verbose,
        )
        
        time_elapsed = time.time() - start_time
        
        # Calculate throughput
        num_tokens = results.get('num_batches', 0) * dataloader.batch_size
        tokens_per_second = num_tokens / time_elapsed if time_elapsed > 0 else 0.0
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            benchmark_name="language_modeling",
            dataset_name=dataset_name,
            num_samples=results.get('num_batches', 0),
            perplexity=results.get('perplexity', float('inf')),
            accuracy=results.get('accuracy', 0.0),
            loss=results.get('loss', 0.0),
            time_elapsed=time_elapsed,
            tokens_per_second=tokens_per_second,
        )
        
        if verbose:
            print(benchmark_result.summary())
        
        return benchmark_result
    
    def compare_checkpoints(
        self,
        checkpoint_paths: List[Path],
        dataloader,
        dataset_name: str = "custom",
    ) -> List[BenchmarkResult]:
        """
        Compare multiple model checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint files
            dataloader: Test dataloader
            dataset_name: Dataset name
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for checkpoint_path in checkpoint_paths:
            print(f"\nEvaluating checkpoint: {checkpoint_path.name}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Run benchmark
            result = self.run_benchmark(
                dataloader,
                dataset_name=f"{dataset_name}_{checkpoint_path.stem}",
                verbose=False,
            )
            results.append(result)
            
            print(f"  Perplexity: {result.perplexity:.2f}")
            print(f"  Accuracy: {result.accuracy:.2f}%")
        
        # Find best checkpoint
        best_result = min(results, key=lambda r: r.perplexity)
        print(f"\nBest checkpoint: {best_result.dataset_name}")
        print(f"  Perplexity: {best_result.perplexity:.2f}")
        
        return results


class DomainBenchmark:
    """
    Domain-specific benchmarks for NOVA.
    
    Evaluates model on specialized tasks:
    - Physics knowledge
    - Mathematical reasoning
    - Code understanding
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
    ):
        """
        Initialize domain benchmark.
        
        Args:
            model: NOVA model
            tokenizer: Tokenizer
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def run_physics_benchmark(
        self,
        questions: List[str],
        answers: List[str],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate physics knowledge.
        
        Args:
            questions: Physics questions
            answers: Correct answers
            verbose: Print progress
            
        Returns:
            Accuracy metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Running Physics Benchmark")
            print(f"{'='*60}")
        
        correct = 0
        total = len(questions)
        
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Question {idx + 1}/{total}")
            
            # Generate prediction
            prediction = self._generate_answer(question)
            
            # Check if correct (simple string matching for now)
            if self._check_answer(prediction, answer):
                correct += 1
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        
        if verbose:
            print(f"\nPhysics Benchmark Results:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Correct: {correct}/{total}")
        
        return results
    
    def run_math_benchmark(
        self,
        problems: List[str],
        solutions: List[str],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate mathematical reasoning.
        
        Args:
            problems: Math problems
            solutions: Correct solutions
            verbose: Print progress
            
        Returns:
            Accuracy metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Running Math Benchmark")
            print(f"{'='*60}")
        
        correct = 0
        total = len(problems)
        
        for idx, (problem, solution) in enumerate(zip(problems, solutions)):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Problem {idx + 1}/{total}")
            
            # Generate prediction
            prediction = self._generate_answer(problem)
            
            # Check if correct
            if self._check_answer(prediction, solution):
                correct += 1
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        
        if verbose:
            print(f"\nMath Benchmark Results:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Correct: {correct}/{total}")
        
        return results
    
    def run_code_benchmark(
        self,
        tasks: List[Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate code understanding and generation.
        
        Args:
            tasks: List of coding tasks with 'prompt' and 'expected_output'
            verbose: Print progress
            
        Returns:
            Success metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Running Code Benchmark")
            print(f"{'='*60}")
        
        correct = 0
        total = len(tasks)
        
        for idx, task in enumerate(tasks):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Task {idx + 1}/{total}")
            
            prompt = task['prompt']
            expected = task.get('expected_output', '')
            
            # Generate code
            generated_code = self._generate_answer(prompt)
            
            # Evaluate (could use execution or string matching)
            if self._check_answer(generated_code, expected):
                correct += 1
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        
        if verbose:
            print(f"\nCode Benchmark Results:")
            print(f"  Success Rate: {accuracy:.2f}%")
            print(f"  Correct: {correct}/{total}")
        
        return results
    
    @torch.no_grad()
    def _generate_answer(self, prompt: str, max_length: int = 64) -> str:
        """Generate answer for prompt."""
        self.model.eval()
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).to(self.device)
        
        # Generate
        generated = input_ids
        for _ in range(max_length):
            outputs = self.model(generated)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS if available
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        answer = self.tokenizer.decode(generated[0].tolist())
        
        # Remove prompt from answer
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        
        return answer
    
    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        Simple implementation - can be enhanced with:
        - Fuzzy matching
        - Semantic similarity
        - Custom domain logic
        """
        # Normalize
        pred = prediction.lower().strip()
        truth = ground_truth.lower().strip()
        
        # Exact match
        if pred == truth:
            return True
        
        # Substring match (for partial credit)
        if truth in pred or pred in truth:
            return True
        
        return False
    
    def save_benchmark_results(
        self,
        results: Dict[str, Any],
        filepath: Path,
    ):
        """Save benchmark results to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for NOVA.
    
    Runs multiple benchmarks and aggregates results.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
    ):
        """
        Initialize benchmark suite.
        
        Args:
            model: NOVA model
            tokenizer: Tokenizer
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.lm_benchmark = LanguageModelingBenchmark(model, device)
        self.domain_benchmark = DomainBenchmark(model, tokenizer, device)
    
    def run_all_benchmarks(
        self,
        lm_dataloader=None,
        physics_data: Optional[Tuple[List[str], List[str]]] = None,
        math_data: Optional[Tuple[List[str], List[str]]] = None,
        code_data: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all available benchmarks.
        
        Args:
            lm_dataloader: Language modeling dataloader
            physics_data: (questions, answers) for physics
            math_data: (problems, solutions) for math
            code_data: Coding tasks
            verbose: Print progress
            
        Returns:
            Aggregated benchmark results
        """
        results = {}
        
        # Language modeling
        if lm_dataloader:
            lm_result = self.lm_benchmark.run_benchmark(
                lm_dataloader,
                dataset_name="general",
                verbose=verbose,
            )
            results['language_modeling'] = lm_result.to_dict()
        
        # Physics
        if physics_data:
            questions, answers = physics_data
            physics_results = self.domain_benchmark.run_physics_benchmark(
                questions, answers, verbose=verbose
            )
            results['physics'] = physics_results
        
        # Math
        if math_data:
            problems, solutions = math_data
            math_results = self.domain_benchmark.run_math_benchmark(
                problems, solutions, verbose=verbose
            )
            results['math'] = math_results
        
        # Code
        if code_data:
            code_results = self.domain_benchmark.run_code_benchmark(
                code_data, verbose=verbose
            )
            results['code'] = code_results
        
        return results
    
    def save_all_results(self, results: Dict[str, Any], filepath: Path):
        """Save all benchmark results."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAll benchmark results saved to {filepath}")
