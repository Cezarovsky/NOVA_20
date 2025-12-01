"""
📄 NOVA Document Chunker

Intelligent document chunking for RAG:
- Split documents into manageable chunks
- Preserve context with overlap
- Handle different document types (text, PDF, code)
- Maintain metadata through chunking
"""

from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import PDF support
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. PDF support disabled.")


class DocumentChunker:
    """
    Smart document chunking with overlap and metadata preservation.
    
    Supports multiple chunking strategies:
    - Fixed size chunks
    - Sentence-based chunks
    - Paragraph-based chunks
    - Code-aware chunks
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = 'smart'  # 'fixed', 'sentence', 'paragraph', 'smart', 'code'
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        logger.info(f"✓ Document chunker initialized")
        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Overlap: {chunk_overlap}")
        logger.info(f"  Strategy: {strategy}")
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to all chunks
            
        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if metadata is None:
            metadata = {}
        
        # Choose chunking strategy
        if self.strategy == 'fixed':
            chunks = self._chunk_fixed(text)
        elif self.strategy == 'sentence':
            chunks = self._chunk_sentences(text)
        elif self.strategy == 'paragraph':
            chunks = self._chunk_paragraphs(text)
        elif self.strategy == 'code':
            chunks = self._chunk_code(text)
        else:  # 'smart'
            chunks = self._chunk_smart(text)
        
        # Add metadata and chunk index
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            chunk_metadata['chunk_strategy'] = self.strategy
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        logger.info(f"✓ Chunked text into {len(result)} chunks")
        return result
    
    def _chunk_fixed(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Sentence-based chunking."""
        # Split into sentences (simple regex)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (last few sentences)
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_paragraphs(self, text: str) -> List[str]:
        """Paragraph-based chunking."""
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_code(self, text: str) -> List[str]:
        """Code-aware chunking (preserves functions/classes)."""
        # Split by function/class definitions
        patterns = [
            r'\n(?=def\s+\w+)',      # Python functions
            r'\n(?=class\s+\w+)',    # Python classes
            r'\n(?=function\s+\w+)', # JavaScript functions
            r'\n(?=\w+\s+\w+\s*\()', # C-style functions
        ]
        
        # Try to split by code blocks
        blocks = [text]
        for pattern in patterns:
            new_blocks = []
            for block in blocks:
                new_blocks.extend(re.split(pattern, block))
            blocks = new_blocks
        
        # Group small blocks together
        chunks = []
        current_chunk = []
        current_length = 0
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            block_length = len(block)
            
            if current_length + block_length > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(block)
            current_length += block_length
        
        # Add last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _chunk_smart(self, text: str) -> List[str]:
        """
        Smart chunking that detects text type and uses appropriate strategy.
        """
        # Detect if text looks like code
        code_indicators = ['def ', 'class ', 'function ', 'import ', '#include']
        is_code = any(indicator in text for indicator in code_indicators)
        
        if is_code:
            return self._chunk_code(text)
        
        # Check if text has clear paragraph structure
        paragraph_count = len(re.findall(r'\n\s*\n', text))
        has_paragraphs = paragraph_count > 2
        
        if has_paragraphs:
            return self._chunk_paragraphs(text)
        else:
            return self._chunk_sentences(text)
    
    def chunk_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk a file based on its type.
        
        Args:
            file_path: Path to file
            metadata: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        path = Path(file_path)
        
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata['source'] = str(path)
        metadata['filename'] = path.name
        metadata['extension'] = path.suffix
        
        # Handle different file types
        if path.suffix == '.pdf' and PDF_AVAILABLE:
            text = self._read_pdf(file_path)
        elif path.suffix in ['.txt', '.md', '.py', '.js', '.java', '.cpp', '.h']:
            text = path.read_text(encoding='utf-8')
        else:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return []
        
        return self.chunk_text(text, metadata)
    
    def _read_pdf(self, file_path: str) -> str:
        """Read text from PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF support")
        
        reader = PdfReader(file_path)
        text = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return '\n\n'.join(text)
    
    def chunk_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            
        Returns:
            Tuple of (chunked_texts, chunked_metadatas)
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        all_chunks = []
        all_metadatas = []
        
        for doc, meta in zip(documents, metadatas):
            chunks = self.chunk_text(doc, meta)
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                all_metadatas.append(chunk['metadata'])
        
        logger.info(f"✓ Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks, all_metadatas


if __name__ == "__main__":
    # Demo
    print("📄 Document Chunker Demo\n")
    
    # Create chunker
    chunker = DocumentChunker(
        chunk_size=200,
        chunk_overlap=50,
        strategy='smart'
    )
    
    # Test text
    text = """
    NOVA is an intelligent AI assistant built with advanced transformer architecture.
    It provides natural language understanding and generation capabilities.
    
    The system includes multiple components working together. The core transformer
    handles language processing. The RAG system provides long-term memory.
    The voice module enables spoken interaction.
    
    NOVA can be used for various tasks including question answering, text generation,
    and conversational AI. It supports multiple languages including English and Romanian.
    """
    
    # Chunk text
    chunks = chunker.chunk_text(text, metadata={'source': 'demo'})
    
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Text: {chunk['text'][:100]}...")
        print(f"  Metadata: {chunk['metadata']}\n")
    
    print("✓ Chunking demo complete!")
