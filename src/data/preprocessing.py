"""
Data Preprocessing for NOVA

Text cleaning, normalization, chunking, and quality filtering.
"""

import re
import unicodedata
from typing import List, Optional, Tuple
import numpy as np


class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    
    Handles:
    - Unicode normalization
    - Whitespace cleaning
    - Special character handling
    - Case normalization (optional)
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        preserve_math_symbols: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_unicode: Normalize unicode characters
            preserve_math_symbols: Keep math/physics symbols
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.preserve_math_symbols = preserve_math_symbols
    
    def __call__(self, text: str) -> str:
        """Preprocess text."""
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Lowercase
        if self.lowercase and not self.preserve_math_symbols:
            text = text.lower()
        elif self.lowercase and self.preserve_math_symbols:
            # Lowercase but preserve special symbols
            result = []
            for char in text:
                if char in '∇∂∫∮∑∏αβγδεθλμνπρστφχψωΔΓΘΛΠΣΦΨΩℏℓ∞∝√∛∜≈≠≤≥≡∈∉⊂⊃∪∩∀∃∄∅ℕℤℚℝℂ→←↔⇒⇐⇔':
                    result.append(char)
                else:
                    result.append(char.lower())
            text = ''.join(result)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
        return [self(text) for text in texts]


class DataCleaner:
    """
    Advanced data cleaning.
    
    Removes low-quality or problematic text:
    - Too short/long
    - High repetition
    - Low entropy
    - Bad character ratios
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        max_repetition_ratio: float = 0.5,
        min_entropy: float = 3.0,
        max_special_char_ratio: float = 0.3,
    ):
        """
        Initialize cleaner.
        
        Args:
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
            max_repetition_ratio: Max ratio of most common n-gram
            min_entropy: Minimum character entropy
            max_special_char_ratio: Max ratio of special characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
        self.min_entropy = min_entropy
        self.max_special_char_ratio = max_special_char_ratio
    
    def is_valid(self, text: str) -> bool:
        """Check if text passes quality filters."""
        # Length check
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Repetition check
        if self._repetition_ratio(text) > self.max_repetition_ratio:
            return False
        
        # Entropy check
        if self._char_entropy(text) < self.min_entropy:
            return False
        
        # Special character ratio
        if self._special_char_ratio(text) > self.max_special_char_ratio:
            return False
        
        return True
    
    def _repetition_ratio(self, text: str, n: int = 3) -> float:
        """Calculate repetition ratio for n-grams."""
        if len(text) < n:
            return 0.0
        
        # Count n-grams
        ngrams = {}
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
        if not ngrams:
            return 0.0
        
        # Find most common
        max_count = max(ngrams.values())
        total_count = len(text) - n + 1
        
        return max_count / total_count
    
    def _char_entropy(self, text: str) -> float:
        """Calculate character entropy."""
        if not text:
            return 0.0
        
        # Character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy
    
    def _special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters."""
        if not text:
            return 0.0
        
        special_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return special_count / len(text)
    
    def filter_batch(self, texts: List[str]) -> List[str]:
        """Filter batch of texts."""
        return [text for text in texts if self.is_valid(text)]


class TextChunker:
    """
    Chunk long texts into smaller pieces.
    
    Useful for processing documents longer than model's context window.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        respect_sentences: bool = True,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size (characters or tokens)
            overlap: Overlap between chunks
            respect_sentences: Try to chunk at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text into pieces."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        if self.respect_sentences:
            # Split into sentences
            sentences = self._split_sentences(text)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length <= self.chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Save current chunk
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and current_chunk:
                        # Include last few sentences for overlap
                        overlap_text = ' '.join(current_chunk[-2:])
                        if len(overlap_text) <= self.overlap:
                            current_chunk = current_chunk[-2:]
                            current_length = len(overlap_text)
                        else:
                            current_chunk = []
                            current_length = 0
                    else:
                        current_chunk = []
                        current_length = 0
                    
                    # Add current sentence
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        else:
            # Simple character-based chunking
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - self.overlap
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class QualityFilter:
    """
    Domain-specific quality filtering.
    
    Checks for domain-specific quality indicators:
    - Physics: equations, symbols, units
    - Math: proofs, theorems, notation
    - Code: syntax, structure, comments
    """
    
    def __init__(self, domain: str = 'general'):
        """
        Initialize quality filter.
        
        Args:
            domain: Target domain (physics, math, code, general)
        """
        self.domain = domain
    
    def score(self, text: str) -> float:
        """
        Score text quality for domain.
        
        Returns:
            Quality score (0.0 - 1.0)
        """
        if self.domain == 'physics':
            return self._score_physics(text)
        elif self.domain == 'math':
            return self._score_math(text)
        elif self.domain == 'code':
            return self._score_code(text)
        else:
            return self._score_general(text)
    
    def _score_physics(self, text: str) -> float:
        """Score physics text quality."""
        score = 0.0
        
        # Check for physics symbols
        physics_symbols = ['∇', '∂', '∫', '∑', '∏', 'α', 'β', 'γ', 'θ', 'λ', 'π', 'σ', 'φ', 'ψ', 'ω']
        symbol_count = sum(1 for sym in physics_symbols if sym in text)
        score += min(symbol_count * 0.1, 0.3)
        
        # Check for equations
        equation_patterns = [r'\d+\s*[+\-*/=]\s*\d+', r'[a-z]\s*=\s*[a-z0-9]']
        for pattern in equation_patterns:
            if re.search(pattern, text):
                score += 0.2
                break
        
        # Check for units
        units = ['m', 'kg', 's', 'J', 'N', 'W', 'V', 'A', 'Hz']
        unit_count = sum(1 for unit in units if f' {unit}' in text or f'{unit}/' in text)
        score += min(unit_count * 0.05, 0.2)
        
        # Check for physics keywords
        keywords = ['energy', 'force', 'mass', 'velocity', 'acceleration', 'momentum', 'wave', 'field']
        keyword_count = sum(1 for kw in keywords if kw in text.lower())
        score += min(keyword_count * 0.05, 0.3)
        
        return min(score, 1.0)
    
    def _score_math(self, text: str) -> float:
        """Score math text quality."""
        score = 0.0
        
        # Check for math symbols
        math_symbols = ['√', '∑', '∏', '∫', '≈', '≠', '≤', '≥', '∈', '⊂', '∩', '∪']
        symbol_count = sum(1 for sym in math_symbols if sym in text)
        score += min(symbol_count * 0.1, 0.3)
        
        # Check for theorems/proofs
        if re.search(r'\btheorem\b|\bproof\b|\blemma\b|\bcorollary\b', text, re.IGNORECASE):
            score += 0.3
        
        # Check for mathematical notation
        if re.search(r'\b[a-z]\s*=\s*[a-z0-9]', text):
            score += 0.2
        
        # Check for math keywords
        keywords = ['function', 'derivative', 'integral', 'limit', 'matrix', 'vector', 'equation']
        keyword_count = sum(1 for kw in keywords if kw in text.lower())
        score += min(keyword_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _score_code(self, text: str) -> float:
        """Score code text quality."""
        score = 0.0
        
        # Check for code keywords
        keywords = ['def', 'class', 'return', 'if', 'else', 'for', 'while', 'import']
        keyword_count = sum(1 for kw in keywords if f' {kw} ' in text or f'\n{kw} ' in text)
        score += min(keyword_count * 0.1, 0.4)
        
        # Check for code structure
        if re.search(r'def\s+\w+\s*\(', text):  # Function definition
            score += 0.2
        if re.search(r'class\s+\w+', text):  # Class definition
            score += 0.2
        
        # Check for indentation
        if re.search(r'\n\s{4,}', text):  # Indented blocks
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_general(self, text: str) -> float:
        """Score general text quality."""
        score = 0.5  # Base score
        
        # Check sentence structure
        sentence_count = len(re.findall(r'[.!?]', text))
        if sentence_count > 0:
            score += 0.2
        
        # Check capitalization
        if any(char.isupper() for char in text):
            score += 0.1
        
        # Check punctuation
        if any(char in text for char in ',.;:'):
            score += 0.2
        
        return min(score, 1.0)
    
    def filter_batch(self, texts: List[str], min_score: float = 0.5) -> List[str]:
        """Filter batch by minimum quality score."""
        return [text for text in texts if self.score(text) >= min_score]
