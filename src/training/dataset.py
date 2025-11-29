"""
NOVA Dataset & DataLoader

Handles training data for NOVA with AI2AI embedding support.
Supports both text-based and pre-computed embedding datasets.
"""

from typing import List, Dict, Any, Optional, Iterator, Tuple
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from pathlib import Path
import json

from ..ai2ai.protocol import AI2AIMessage, MessageType


class NovaDataset(Dataset):
    """
    Dataset for NOVA training.
    
    Supports two modes:
    1. Text mode: Raw text strings (will be tokenized)
    2. Embedding mode: Pre-computed AI2AI embeddings from Claude
    """
    
    def __init__(
        self,
        data: List[str] | List[AI2AIMessage] | List[torch.Tensor],
        mode: str = "text",
        max_seq_length: int = 512,
        embedding_dim: int = 768,
        vocab_size: int = 50000,
    ):
        """
        Initialize dataset.
        
        Args:
            data: Training data (text, AI2AI messages, or embeddings)
            mode: "text", "ai2ai", or "embeddings"
            max_seq_length: Maximum sequence length
            embedding_dim: Embedding dimension
            vocab_size: Vocabulary size (for text mode)
        """
        self.data = data
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Validate mode
        if mode not in ["text", "ai2ai", "embeddings"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'text', 'ai2ai', or 'embeddings'")
        
        # Validate data types
        if mode == "text" and data and not isinstance(data[0], str):
            raise TypeError("Text mode requires string data")
        elif mode == "ai2ai" and data and not isinstance(data[0], AI2AIMessage):
            raise TypeError("AI2AI mode requires AI2AIMessage data")
        elif mode == "embeddings" and data and not isinstance(data[0], torch.Tensor):
            raise TypeError("Embeddings mode requires torch.Tensor data")
    
    def __len__(self) -> int:
        """Dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs [seq_len] (text mode)
            - embeddings: Embedding vectors [seq_len, emb_dim] (embedding mode)
            - attention_mask: Attention mask [seq_len]
            - labels: Target tokens for next-token prediction [seq_len]
        """
        item = self.data[idx]
        
        if self.mode == "text":
            return self._process_text(item)
        elif self.mode == "ai2ai":
            return self._process_ai2ai(item)
        elif self.mode == "embeddings":
            return self._process_embeddings(item)
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text into training sample."""
        # Simple tokenization (placeholder - real tokenizer would be more sophisticated)
        tokens = self._tokenize(text)
        
        # Truncate or pad to max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        
        # Pad if needed
        if len(tokens) < self.max_seq_length:
            padding_length = self.max_seq_length - len(tokens)
            tokens.extend([0] * padding_length)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels for next-token prediction (shifted by 1)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = 0  # Padding for last position
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _process_ai2ai(self, message: AI2AIMessage) -> Dict[str, torch.Tensor]:
        """Process AI2AI message into training sample."""
        embeddings = message.embeddings
        
        # Handle shape: should be [seq_len, emb_dim]
        if embeddings.dim() == 1:
            # Single embedding vector - expand to sequence
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            # Flatten to 2D
            embeddings = embeddings.view(-1, embeddings.size(-1))
        
        seq_len = embeddings.size(0)
        
        # Truncate or pad
        if seq_len > self.max_seq_length:
            embeddings = embeddings[:self.max_seq_length]
            seq_len = self.max_seq_length
        
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        # Pad if needed
        if seq_len < self.max_seq_length:
            padding_length = self.max_seq_length - seq_len
            padding = torch.zeros(padding_length, self.embedding_dim)
            embeddings = torch.cat([embeddings, padding], dim=0)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        # Labels for next-embedding prediction
        labels = embeddings.clone()
        labels[:-1] = embeddings[1:]  # Shift left
        labels[-1] = 0  # Padding for last position
        
        return {
            "embeddings": embeddings,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _process_embeddings(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process raw embeddings into training sample."""
        # Similar to AI2AI processing
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            embeddings = embeddings.view(-1, embeddings.size(-1))
        
        seq_len = embeddings.size(0)
        
        if seq_len > self.max_seq_length:
            embeddings = embeddings[:self.max_seq_length]
            seq_len = self.max_seq_length
        
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        if seq_len < self.max_seq_length:
            padding_length = self.max_seq_length - seq_len
            padding = torch.zeros(padding_length, embeddings.size(-1))
            embeddings = torch.cat([embeddings, padding], dim=0)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        labels = embeddings.clone()
        labels[:-1] = embeddings[1:]
        labels[-1] = 0
        
        return {
            "embeddings": embeddings,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Simple tokenization (placeholder).
        
        TODO: Replace with proper tokenizer (tiktoken, sentencepiece, etc.)
        """
        # Character-level for now
        tokens = [ord(c) % self.vocab_size for c in text]
        return tokens
    
    @classmethod
    def from_file(cls, filepath: Path, mode: str = "text", **kwargs) -> 'NovaDataset':
        """
        Load dataset from file.
        
        Supports:
        - .txt: Text file (one sample per line)
        - .json: JSON array of strings
        - .jsonl: JSONL format
        - .pt: Saved torch tensors
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        if filepath.suffix == ".txt":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f if line.strip()]
        
        elif filepath.suffix == ".json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        elif filepath.suffix == ".jsonl":
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
        
        elif filepath.suffix == ".pt":
            data = torch.load(filepath)
            if isinstance(data, torch.Tensor):
                # Split into chunks
                data = list(torch.split(data, kwargs.get('chunk_size', 512)))
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(data, mode=mode, **kwargs)


class NovaDataLoader:
    """
    Custom DataLoader for NOVA training.
    
    Wraps PyTorch DataLoader with NOVA-specific features:
    - AI2AI embedding batching
    - Dynamic padding
    - Mixed text/embedding batches
    """
    
    def __init__(
        self,
        dataset: NovaDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize DataLoader.
        
        Args:
            dataset: NovaDataset instance
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Create PyTorch DataLoader
        self.dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch samples.
        
        Stacks tensors and handles mixed types.
        """
        if not batch:
            return {}
        
        # Check what keys are present
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            tensors = [item[key] for item in batch]
            collated[key] = torch.stack(tensors)
        
        return collated
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Number of batches."""
        return len(self.dataloader)


class StreamingDataset:
    """
    Streaming dataset for very large corpora.
    
    Doesn't load all data into memory - streams from disk/network.
    Useful for training on massive text corpora.
    """
    
    def __init__(
        self,
        data_source: Path | str,
        chunk_size: int = 1024,
        buffer_size: int = 10000,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_source: Path to data file or directory
            chunk_size: Size of chunks to read
            buffer_size: Number of samples to buffer
        """
        self.data_source = Path(data_source)
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        if not self.data_source.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")
    
    def stream(self) -> Iterator[str]:
        """
        Stream data samples.
        
        Yields:
            Text samples one at a time
        """
        if self.data_source.is_file():
            yield from self._stream_file(self.data_source)
        elif self.data_source.is_dir():
            for file in sorted(self.data_source.glob("*.txt")):
                yield from self._stream_file(file)
    
    def _stream_file(self, filepath: Path) -> Iterator[str]:
        """Stream samples from a single file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            buffer = []
            for line in f:
                line = line.strip()
                if line:
                    buffer.append(line)
                    if len(buffer) >= self.buffer_size:
                        for sample in buffer:
                            yield sample
                        buffer = []
            
            # Yield remaining buffer
            for sample in buffer:
                yield sample
