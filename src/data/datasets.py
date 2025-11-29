"""
Dataset Classes for NOVA

Efficient dataset implementations with:
- Memory mapping for large datasets
- Caching for fast access
- Domain-specific datasets
- Multi-domain mixing
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path
import json
import pickle
import mmap
import numpy as np
from collections import defaultdict


class TextDataset(Dataset):
    """
    Simple text dataset.
    
    Loads texts from file or list and tokenizes on-the-fly.
    """
    
    def __init__(
        self,
        data: Union[List[str], str, Path],
        tokenizer: 'NovaTokenizer',
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize text dataset.
        
        Args:
            data: List of texts or path to text file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            cache_dir: Directory for caching tokenized data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load data
        if isinstance(data, (str, Path)):
            data = Path(data)
            with open(data, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
        else:
            self.texts = data
        
        print(f"Loaded {len(self.texts)} texts")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example."""
        text = self.texts[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Create tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text,
        }


class DomainDataset(Dataset):
    """
    Domain-specific dataset.
    
    Associates each example with a domain label (physics, math, code).
    """
    
    DOMAINS = ['physics', 'math', 'code', 'general']
    
    def __init__(
        self,
        data: Union[List[Dict], str, Path],
        tokenizer: 'NovaTokenizer',
        domain: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize domain dataset.
        
        Args:
            data: List of dicts with 'text' and 'domain' keys, or path to JSONL file
            tokenizer: Tokenizer instance
            domain: Filter to specific domain (if None, use all)
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain = domain
        
        # Load data
        if isinstance(data, (str, Path)):
            data = Path(data)
            self.examples = []
            
            with open(data, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        if domain is None or example.get('domain') == domain:
                            self.examples.append(example)
        else:
            if domain is None:
                self.examples = data
            else:
                self.examples = [ex for ex in data if ex.get('domain') == domain]
        
        # Domain to ID mapping
        self.domain_to_id = {d: i for i, d in enumerate(self.DOMAINS)}
        
        print(f"Loaded {len(self.examples)} examples" + 
              (f" for domain '{domain}'" if domain else ""))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example with domain label."""
        example = self.examples[idx]
        text = example['text']
        domain = example.get('domain', 'general')
        
        # Tokenize
        token_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Create tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        domain_id = torch.tensor(self.domain_to_id.get(domain, 3), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'domain_id': domain_id,
            'domain': domain,
            'text': text,
        }


class MultiDomainDataset(Dataset):
    """
    Multi-domain dataset with balanced sampling.
    
    Combines multiple domain datasets with configurable mixing ratios.
    """
    
    def __init__(
        self,
        domain_datasets: Dict[str, DomainDataset],
        mixing_ratios: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize multi-domain dataset.
        
        Args:
            domain_datasets: Dict of domain name -> DomainDataset
            mixing_ratios: Sampling probability per domain (if None, uniform)
            temperature: Temperature for ratio smoothing (1.0 = as-is, <1.0 = more uniform)
        """
        self.domain_datasets = domain_datasets
        self.domains = list(domain_datasets.keys())
        
        # Calculate dataset sizes
        self.domain_sizes = {
            domain: len(dataset)
            for domain, dataset in domain_datasets.items()
        }
        
        # Set mixing ratios
        if mixing_ratios is None:
            # Uniform mixing
            mixing_ratios = {domain: 1.0 for domain in self.domains}
        
        # Apply temperature
        total = sum(mixing_ratios.values())
        self.mixing_ratios = {
            domain: (ratio / total) ** (1.0 / temperature)
            for domain, ratio in mixing_ratios.items()
        }
        
        # Normalize
        total = sum(self.mixing_ratios.values())
        self.mixing_ratios = {
            domain: ratio / total
            for domain, ratio in self.mixing_ratios.items()
        }
        
        # Create sampling indices
        self._create_sampling_indices()
        
        print(f"Multi-domain dataset:")
        for domain in self.domains:
            print(f"  {domain}: {self.domain_sizes[domain]} examples "
                  f"(ratio: {self.mixing_ratios[domain]:.2%})")
    
    def _create_sampling_indices(self):
        """Create indices for balanced sampling."""
        # Calculate number of samples per domain based on ratios
        total_samples = sum(self.domain_sizes.values())
        
        self.samples_per_domain = {
            domain: int(total_samples * self.mixing_ratios[domain])
            for domain in self.domains
        }
        
        # Create index mapping: global_idx -> (domain, local_idx)
        self.index_mapping = []
        
        for domain in self.domains:
            domain_size = self.domain_sizes[domain]
            num_samples = self.samples_per_domain[domain]
            
            # Sample with replacement if needed
            if num_samples > domain_size:
                indices = np.random.choice(domain_size, num_samples, replace=True)
            else:
                indices = np.random.choice(domain_size, num_samples, replace=False)
            
            for idx in indices:
                self.index_mapping.append((domain, int(idx)))
        
        # Shuffle
        np.random.shuffle(self.index_mapping)
    
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example from appropriate domain dataset."""
        domain, local_idx = self.index_mapping[idx]
        return self.domain_datasets[domain][local_idx]
    
    def resample(self):
        """Resample indices (call between epochs for variety)."""
        self._create_sampling_indices()


class CachedDataset(Dataset):
    """
    Cached dataset for fast loading.
    
    Preprocesses and caches tokenized examples to disk.
    Useful for large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        source_dataset: Dataset,
        cache_file: Union[str, Path],
        rebuild_cache: bool = False,
    ):
        """
        Initialize cached dataset.
        
        Args:
            source_dataset: Original dataset to cache
            cache_file: Path to cache file
            rebuild_cache: Force rebuild cache
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build or load cache
        if rebuild_cache or not self.cache_file.exists():
            self._build_cache(source_dataset)
        
        # Load cache
        with open(self.cache_file, 'rb') as f:
            self.cached_data = pickle.load(f)
        
        print(f"Loaded {len(self.cached_data)} cached examples from {self.cache_file}")
    
    def _build_cache(self, source_dataset: Dataset):
        """Build cache from source dataset."""
        print(f"Building cache for {len(source_dataset)} examples...")
        
        cached_data = []
        for i in range(len(source_dataset)):
            example = source_dataset[i]
            cached_data.append(example)
            
            if (i + 1) % 10000 == 0:
                print(f"  Cached {i + 1}/{len(source_dataset)}")
        
        # Save cache
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f"✓ Cache saved to {self.cache_file}")
    
    def __len__(self) -> int:
        return len(self.cached_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.cached_data[idx]


class MemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for very large datasets.
    
    Uses memory mapping to avoid loading entire dataset into RAM.
    Efficient for datasets too large for memory.
    """
    
    def __init__(
        self,
        data_file: Union[str, Path],
        index_file: Union[str, Path],
        tokenizer: 'NovaTokenizer',
        max_length: int = 512,
    ):
        """
        Initialize memory-mapped dataset.
        
        Args:
            data_file: Path to binary data file
            index_file: Path to index file (byte offsets)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data_file = Path(data_file)
        self.index_file = Path(index_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load index
        with open(self.index_file, 'rb') as f:
            self.index = pickle.load(f)
        
        # Open memory-mapped file
        self.mmap_file = open(self.data_file, 'rb')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
        
        print(f"Memory-mapped dataset: {len(self.index)} examples")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example from memory-mapped file."""
        # Get byte offset from index
        offset, length = self.index[idx]
        
        # Read from mmap
        self.mmap.seek(offset)
        data = self.mmap.read(length)
        
        # Deserialize
        example = pickle.loads(data)
        
        # Tokenize if needed
        if 'input_ids' not in example:
            token_ids = self.tokenizer.encode(
                example['text'],
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
            )
            
            example['input_ids'] = torch.tensor(token_ids, dtype=torch.long)
            example['attention_mask'] = (
                example['input_ids'] != self.tokenizer.pad_token_id
            ).long()
        
        return example
    
    def __del__(self):
        """Close memory-mapped file."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'mmap_file'):
            self.mmap_file.close()


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for infinite data generation.
    
    Useful for online learning or when dataset doesn't fit on disk.
    """
    
    def __init__(
        self,
        data_generator: Callable,
        tokenizer: 'NovaTokenizer',
        max_length: int = 512,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_generator: Generator function that yields text examples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data_generator = data_generator
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        """Iterate over generated examples."""
        for text in self.data_generator():
            # Tokenize
            token_ids = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
            )
            
            # Create tensors
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'text': text,
            }
