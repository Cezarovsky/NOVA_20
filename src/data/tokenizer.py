"""
NOVA Tokenizer

BPE-based tokenizer with domain-specific vocabulary:
- Physics symbols: ∇, ∂, ∫, Σ, etc.
- Math operators: √, ∞, ≈, ≠, etc.
- Code tokens: def, class, return, etc.
- Special tokens: [PAD], [BOS], [EOS], [UNK], [MASK]
"""

import json
import re
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np


class NovaTokenizer:
    """
    BPE tokenizer for NOVA with domain-specific vocabulary.
    
    Features:
    - Byte Pair Encoding (BPE) for subword tokenization
    - Special tokens for padding, BOS, EOS, etc.
    - Domain-specific tokens (physics, math, code)
    - Save/load vocabulary
    - Encode/decode with various options
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN = "[MASK]"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    
    # Domain-specific tokens
    PHYSICS_TOKENS = [
        "∇", "∂", "∫", "∮", "∑", "∏",  # Operators
        "α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "ν", "π", "ρ", "σ", "τ", "φ", "χ", "ψ", "ω",  # Greek
        "Δ", "Γ", "Θ", "Λ", "Π", "Σ", "Φ", "Ψ", "Ω",  # Capital Greek
        "ℏ", "ℓ", "∞", "∝",  # Constants
    ]
    
    MATH_TOKENS = [
        "√", "∛", "∜",  # Roots
        "≈", "≠", "≤", "≥", "≡", "∈", "∉", "⊂", "⊃", "∪", "∩",  # Relations
        "∀", "∃", "∄", "∅", "ℕ", "ℤ", "ℚ", "ℝ", "ℂ",  # Sets
        "→", "←", "↔", "⇒", "⇐", "⇔",  # Arrows
    ]
    
    CODE_TOKENS = [
        "def", "class", "return", "if", "else", "elif", "for", "while",
        "import", "from", "as", "try", "except", "finally", "with",
        "lambda", "yield", "async", "await", "assert", "break", "continue",
        "pass", "raise", "None", "True", "False", "and", "or", "not", "in", "is",
    ]
    
    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        add_domain_tokens: bool = True,
    ):
        """
        Initialize NOVA tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for token inclusion
            special_tokens: Additional special tokens
            add_domain_tokens: Include domain-specific tokens
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Initialize vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Add special tokens
        self._special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            self.MASK_TOKEN,
            self.SEP_TOKEN,
            self.CLS_TOKEN,
        ]
        
        if special_tokens:
            self._special_tokens.extend(special_tokens)
        
        # Add domain tokens
        if add_domain_tokens:
            self._special_tokens.extend(self.PHYSICS_TOKENS)
            self._special_tokens.extend(self.MATH_TOKENS)
            self._special_tokens.extend(self.CODE_TOKENS)
        
        # Initialize with special tokens
        for token in self._special_tokens:
            self._add_token(token)
        
        # BPE merge rules (token_pair -> merged_token)
        self.merges: Dict[tuple, str] = {}
        
        # Token frequencies
        self.token_freq: Dict[str, int] = {}
    
    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            return token_id
        return self.token_to_id[token]
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: Training texts
            verbose: Print progress
        """
        if verbose:
            print(f"Training tokenizer on {len(texts)} texts...")
        
        # Tokenize into characters
        words = []
        for text in texts:
            words.extend(text.split())
        
        # Initialize vocabulary with characters
        char_freq: Dict[str, int] = {}
        for word in words:
            for char in word:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Add frequent characters to vocabulary
        for char, freq in char_freq.items():
            if freq >= self.min_frequency:
                self._add_token(char)
        
        # BPE training: iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            # Count all adjacent pairs
            pair_freq: Dict[tuple, int] = {}
            
            for word in words:
                tokens = list(word)
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1
            
            if not pair_freq:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            
            # Merge this pair
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            self._add_token(merged)
            
            # Update words with merged token
            new_words = []
            for word in words:
                new_word = word.replace(
                    best_pair[0] + best_pair[1],
                    merged
                )
                new_words.append(new_word)
            words = new_words
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {best_pair} -> {merged}")
        
        if verbose:
            print(f"✓ Tokenizer trained: {len(self.token_to_id)} tokens")
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply BPE merges to token list."""
        if not self.merges:
            return tokens
        
        while len(tokens) > 1:
            # Find first merge-able pair
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # Find which pairs can be merged
            mergeable = [(i, pair) for i, pair in enumerate(pairs) if pair in self.merges]
            
            if not mergeable:
                break
            
            # Merge first pair
            i, pair = mergeable[0]
            merged = self.merges[pair]
            tokens = tokens[:i] + [merged] + tokens[i + 2:]
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        pad_to_max_length: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add [BOS] and [EOS]
            max_length: Maximum sequence length
            truncation: Truncate if too long
            padding: Pad if too short
            pad_to_max_length: Pad to max_length
            
        Returns:
            List of token IDs
        """
        # Tokenize into characters first
        tokens = []
        for word in text.split():
            word_tokens = list(word)
            word_tokens = self._apply_merges(word_tokens)
            tokens.extend(word_tokens)
            tokens.append(" ")  # Space token
        
        # Remove trailing space
        if tokens and tokens[-1] == " ":
            tokens = tokens[:-1]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_ids.append(
                self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            )
        
        # Add special tokens
        if add_special_tokens:
            token_ids = (
                [self.token_to_id[self.BOS_TOKEN]] +
                token_ids +
                [self.token_to_id[self.EOS_TOKEN]]
            )
        
        # Truncate
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_special_tokens:
                token_ids[-1] = self.token_to_id[self.EOS_TOKEN]
        
        # Pad
        if padding or pad_to_max_length:
            target_length = max_length if pad_to_max_length and max_length else len(token_ids)
            if len(token_ids) < target_length:
                token_ids.extend(
                    [self.token_to_id[self.PAD_TOKEN]] * (target_length - len(token_ids))
                )
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            
            # Skip special tokens if requested
            if skip_special_tokens and token in self._special_tokens:
                continue
            
            tokens.append(token)
        
        # Join tokens
        text = "".join(tokens)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def encode_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text, **kwargs) for text in texts]
    
    def decode_batch(
        self,
        token_ids_batch: List[List[int]],
        **kwargs
    ) -> List[str]:
        """Decode multiple token ID sequences."""
        return [self.decode(token_ids, **kwargs) for token_ids in token_ids_batch]
    
    def save(self, path: Union[str, Path]):
        """
        Save tokenizer to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': {f"{k[0]}_{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self._special_tokens,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NovaTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded tokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            min_frequency=data['min_frequency'],
            special_tokens=[],
            add_domain_tokens=False,
        )
        
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merges = {
            tuple(k.split('_')): v
            for k, v in data['merges'].items()
        }
        tokenizer._special_tokens = data['special_tokens']
        
        print(f"✓ Tokenizer loaded from {path}")
        
        return tokenizer
    
    @property
    def vocab_length(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.token_to_id[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.token_to_id[self.EOS_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.token_to_id[self.UNK_TOKEN]
    
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID."""
        return self.token_to_id[self.MASK_TOKEN]
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_length
    
    def __repr__(self) -> str:
        return f"NovaTokenizer(vocab_size={self.vocab_length})"
