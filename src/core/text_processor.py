"""
Text Processor - Document Chunking and Preprocessing

This module handles text processing for the NOVA AI system:
- Document chunking with overlap
- Text cleaning and normalization
- Metadata extraction
- Token counting
- Multiple chunking strategies

Chunking Strategies:

1. Fixed-size chunking:
   - Split by character/token count
   - Simple and fast
   - Good for uniform documents

2. Semantic chunking:
   - Split by paragraphs/sections
   - Preserves meaning
   - Better for structured text

3. Recursive chunking:
   - Split hierarchically (document → section → paragraph)
   - Most sophisticated
   - Best quality but slower

Architecture:

    Document Input
         ↓
    Text Cleaning
         ↓
    Chunking Strategy
         ↓
    ┌────┴────┐
    │  Chunks │  ← With overlap & metadata
    └─────────┘

Usage:

    # Basic chunking
    processor = TextProcessor()
    chunks = processor.chunk_text(
        text="Long document...",
        chunk_size=1000,
        overlap=200
    )
    
    # With metadata
    chunks = processor.process_document(
        text="Document content",
        metadata={"source": "paper.pdf", "page": 1}
    )
    
    # Semantic chunking
    chunks = processor.chunk_by_paragraphs(
        text="Multi-paragraph text",
        max_chunk_size=1000
    )

Author: NOVA Development Team
Date: 28 November 2025
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    FIXED = "fixed"  # Fixed size chunks
    SEMANTIC = "semantic"  # Paragraph/section based
    RECURSIVE = "recursive"  # Hierarchical splitting


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata
    
    Attributes:
        text: Chunk content
        metadata: Associated metadata
        chunk_id: Unique chunk identifier
        start_char: Starting character position in original text
        end_char: Ending character position
        token_count: Approximate token count
    """
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    
    def __repr__(self) -> str:
        return (
            f"TextChunk(id={self.chunk_id}, "
            f"tokens={self.token_count}, "
            f"chars={len(self.text)})"
        )


class TextProcessor:
    """
    Text processing and chunking utilities
    
    Handles:
    - Text cleaning and normalization
    - Multiple chunking strategies
    - Token counting
    - Metadata extraction
    - Overlap management
    
    Args:
        chunk_size: Default chunk size (characters or tokens)
        chunk_overlap: Default overlap between chunks
        strategy: Default chunking strategy
        min_chunk_size: Minimum chunk size to keep
    
    Example:
        >>> processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
        >>> chunks = processor.chunk_text("Long document...")
        >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED,
        min_chunk_size: int = 50
    ):
        """Initialize text processor"""
        self.settings = get_settings()
        
        # Configuration
        self.chunk_size = chunk_size or self.settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.settings.CHUNK_OVERLAP
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        
        # Validation
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        logger.info(
            f"Initialized TextProcessor: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, strategy={self.strategy}"
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Operations:
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove special characters (optional)
        - Strip leading/trailing whitespace
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks (convert \r\n and \r to \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive line breaks (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        
        Rule of thumb: ~4 characters per token for English text
        More accurate would be using tiktoken or similar
        
        Args:
            text: Text to count
        
        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 chars per token
        return len(text) // 4
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        Chunk text using configured strategy
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            metadata: Metadata to attach to chunks
        
        Returns:
            List of TextChunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        metadata = metadata or {}
        
        # Clean text first
        text = self.clean_text(text)
        
        # Route to appropriate chunking method
        if self.strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed_size(text, chunk_size, chunk_overlap, metadata)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, chunk_size, metadata)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(text, chunk_size, chunk_overlap, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_fixed_size(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """
        Fixed-size chunking with overlap
        
        Algorithm:
        1. Start at position 0
        2. Take chunk_size characters
        3. Move forward by (chunk_size - chunk_overlap)
        4. Repeat until end
        
        Args:
            text: Text to chunk
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            metadata: Metadata
        
        Returns:
            List of chunks
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            
            # Skip if too small
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={**metadata, 'chunk_index': chunk_id},
                    chunk_id=f"chunk_{chunk_id}",
                    start_char=start,
                    end_char=end,
                    token_count=self.estimate_tokens(chunk_text)
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start += chunk_size - chunk_overlap
            
            # Break if we've covered everything
            if end >= text_length:
                break
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _chunk_semantic(
        self,
        text: str,
        max_chunk_size: int,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """
        Semantic chunking by paragraphs
        
        Algorithm:
        1. Split by double newlines (paragraphs)
        2. Group paragraphs to fit max_chunk_size
        3. Try to keep semantic units together
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size
            metadata: Metadata
        
        Returns:
            List of chunks
        """
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        char_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_size = len(paragraph)
            
            # If paragraph alone exceeds max size, split it
            if para_size > max_chunk_size:
                # First, add current accumulated chunk if any
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={**metadata, 'chunk_index': chunk_id, 'type': 'semantic'},
                        chunk_id=f"chunk_{chunk_id}",
                        start_char=char_position - current_size,
                        end_char=char_position,
                        token_count=self.estimate_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(sentence) < self.min_chunk_size:
                        continue
                    
                    chunk = TextChunk(
                        text=sentence,
                        metadata={**metadata, 'chunk_index': chunk_id, 'type': 'semantic'},
                        chunk_id=f"chunk_{chunk_id}",
                        start_char=char_position,
                        end_char=char_position + len(sentence),
                        token_count=self.estimate_tokens(sentence)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    char_position += len(sentence)
                
            # If adding paragraph exceeds max size, save current chunk
            elif current_size + para_size > max_chunk_size:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={**metadata, 'chunk_index': chunk_id, 'type': 'semantic'},
                        chunk_id=f"chunk_{chunk_id}",
                        start_char=char_position - current_size,
                        end_char=char_position,
                        token_count=self.estimate_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with current paragraph
                current_chunk = [paragraph]
                current_size = para_size
            
            # Add paragraph to current chunk
            else:
                current_chunk.append(paragraph)
                current_size += para_size + 2  # +2 for \n\n
            
            char_position += para_size + 2
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={**metadata, 'chunk_index': chunk_id, 'type': 'semantic'},
                    chunk_id=f"chunk_{chunk_id}",
                    start_char=char_position - current_size,
                    end_char=char_position,
                    token_count=self.estimate_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _chunk_recursive(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Dict[str, Any],
        separators: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """
        Recursive hierarchical chunking
        
        Algorithm:
        1. Try splitting by largest separator (e.g., \n\n)
        2. If chunks still too large, try next separator (e.g., \n)
        3. If still too large, try next separator (e.g., '. ')
        4. Finally, fall back to character-level splitting
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            metadata: Metadata
            separators: List of separators (hierarchical)
        
        Returns:
            List of chunks
        """
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ', '']
        
        chunks = []
        chunk_id = 0
        
        def _recursive_split(
            text: str,
            separators: List[str],
            start_pos: int
        ) -> List[TextChunk]:
            """Helper function for recursive splitting"""
            nonlocal chunk_id
            
            if not text or len(text) < self.min_chunk_size:
                return []
            
            # If text fits in chunk_size, return it
            if len(text) <= chunk_size:
                chunk = TextChunk(
                    text=text,
                    metadata={**metadata, 'chunk_index': chunk_id, 'type': 'recursive'},
                    chunk_id=f"chunk_{chunk_id}",
                    start_char=start_pos,
                    end_char=start_pos + len(text),
                    token_count=self.estimate_tokens(text)
                )
                chunk_id += 1
                return [chunk]
            
            # Try current separator
            if not separators:
                # No more separators, do character split
                return self._chunk_fixed_size(text, chunk_size, chunk_overlap, metadata)
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            # Split by current separator
            if separator:
                splits = text.split(separator)
            else:
                # Empty separator means character-level
                return self._chunk_fixed_size(text, chunk_size, chunk_overlap, metadata)
            
            # Recombine splits into appropriate chunks
            result_chunks = []
            current_chunk = []
            current_size = 0
            position = start_pos
            
            for split in splits:
                split_size = len(split)
                
                if split_size > chunk_size:
                    # Split is too large, recurse with next separator
                    if current_chunk:
                        chunk_text = separator.join(current_chunk)
                        result_chunks.extend(
                            _recursive_split(chunk_text, remaining_separators, position - current_size)
                        )
                        current_chunk = []
                        current_size = 0
                    
                    result_chunks.extend(
                        _recursive_split(split, remaining_separators, position)
                    )
                    position += split_size + len(separator)
                
                elif current_size + split_size <= chunk_size:
                    current_chunk.append(split)
                    current_size += split_size + len(separator)
                    position += split_size + len(separator)
                
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunk_text = separator.join(current_chunk)
                        result_chunks.extend(
                            _recursive_split(chunk_text, remaining_separators, position - current_size)
                        )
                    
                    current_chunk = [split]
                    current_size = split_size
                    position += split_size + len(separator)
            
            # Add final chunk
            if current_chunk:
                chunk_text = separator.join(current_chunk)
                result_chunks.extend(
                    _recursive_split(chunk_text, remaining_separators, position - current_size)
                )
            
            return result_chunks
        
        chunks = _recursive_split(text, separators, 0)
        logger.info(f"Created {len(chunks)} recursive chunks")
        return chunks
    
    def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Process complete document (clean + chunk)
        
        High-level convenience method that:
        1. Cleans text
        2. Chunks using configured strategy
        3. Attaches metadata
        
        Args:
            text: Document text
            metadata: Document metadata
            chunk_size: Override chunk size
            chunk_overlap: Override overlap
        
        Returns:
            List of TextChunk objects
        """
        return self.chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata or {}
        )
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunks
        
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'count': 0,
                'total_chars': 0,
                'total_tokens': 0,
                'avg_chars': 0,
                'avg_tokens': 0,
                'min_chars': 0,
                'max_chars': 0
            }
        
        char_counts = [len(c.text) for c in chunks]
        token_counts = [c.token_count for c in chunks]
        
        return {
            'count': len(chunks),
            'total_chars': sum(char_counts),
            'total_tokens': sum(token_counts),
            'avg_chars': sum(char_counts) / len(chunks),
            'avg_tokens': sum(token_counts) / len(chunks),
            'min_chars': min(char_counts),
            'max_chars': max(char_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts)
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TextProcessor(chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, strategy={self.strategy})"
        )


if __name__ == "__main__":
    """Test text processor"""
    print("=" * 80)
    print("Testing Text Processor")
    print("=" * 80)
    
    # Sample text
    sample_text = """
    Artificial Intelligence (AI) is transforming the world. Machine learning, a subset of AI, 
    enables computers to learn from data without explicit programming.
    
    Deep learning uses neural networks with multiple layers. These networks can learn complex 
    patterns and representations from large datasets.
    
    Natural language processing (NLP) allows AI to understand and generate human language. 
    This technology powers chatbots, translation systems, and text analysis tools.
    
    Computer vision enables AI to interpret and understand visual information. Applications 
    include facial recognition, object detection, and autonomous vehicles.
    
    The future of AI looks promising. Advances in quantum computing, neuromorphic chips, and 
    new algorithms continue to push the boundaries of what's possible.
    """
    
    # Test 1: Fixed-size chunking
    print("\n" + "-" * 80)
    print("Test 1: Fixed-Size Chunking")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.FIXED)
    chunks = processor.chunk_text(sample_text)
    
    print(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}:")
        print(f"    Chars: {len(chunk.text)}")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Preview: {chunk.text[:80]}...")
    
    # Test 2: Semantic chunking
    print("\n" + "-" * 80)
    print("Test 2: Semantic Chunking (Paragraphs)")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=300, strategy=ChunkingStrategy.SEMANTIC)
    chunks = processor.chunk_text(sample_text)
    
    print(f"✅ Created {len(chunks)} semantic chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}:")
        print(f"    Chars: {len(chunk.text)}")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Text: {chunk.text[:100]}...")
    
    # Test 3: Recursive chunking
    print("\n" + "-" * 80)
    print("Test 3: Recursive Chunking")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=250, chunk_overlap=50, strategy=ChunkingStrategy.RECURSIVE)
    chunks = processor.chunk_text(sample_text)
    
    print(f"✅ Created {len(chunks)} recursive chunks")
    
    # Test 4: Statistics
    print("\n" + "-" * 80)
    print("Test 4: Chunk Statistics")
    print("-" * 80)
    
    stats = processor.get_chunk_statistics(chunks)
    print(f"✅ Statistics:")
    print(f"    Total chunks: {stats['count']}")
    print(f"    Total characters: {stats['total_chars']}")
    print(f"    Total tokens: {stats['total_tokens']}")
    print(f"    Avg chars/chunk: {stats['avg_chars']:.0f}")
    print(f"    Avg tokens/chunk: {stats['avg_tokens']:.0f}")
    print(f"    Size range: {stats['min_chars']}-{stats['max_chars']} chars")
    
    # Test 5: Process with metadata
    print("\n" + "-" * 80)
    print("Test 5: Process Document with Metadata")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=200, chunk_overlap=50)
    chunks = processor.process_document(
        text=sample_text,
        metadata={"source": "ai_overview.txt", "category": "AI"}
    )
    
    print(f"✅ Processed document into {len(chunks)} chunks")
    print(f"✅ Sample metadata: {chunks[0].metadata}")
    
    print("\n" + "=" * 80)
    print("Text Processor tests completed")
    print("=" * 80)
