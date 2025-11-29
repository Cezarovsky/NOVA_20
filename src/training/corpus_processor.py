"""
Corpus Processor: Convert raw text to AI2AI embeddings via Claude.

This is the ONE-TIME step where Claude helps prepare training data.
After this, NOVA trains independently with no API calls.
"""

from typing import List, Optional, Iterator, Dict, Any
from pathlib import Path
import torch
from tqdm import tqdm
import json
import time

from ..ai2ai.protocol import AI2AIMessage, MessageType, TransferMode, KnowledgeTransfer
from ..ai2ai.claude_adapter import ClaudeAdapter
from ..ai2ai.encoder import AI2AIEncoder
from ..config.settings import get_settings


class CorpusProcessor:
    """
    Process raw text corpus into AI2AI embeddings for NOVA training.
    
    Workflow:
    1. Read raw text corpus
    2. Send to Claude for processing
    3. Claude returns embeddings via AI2AI protocol
    4. Save embeddings for NOVA training
    5. Claude is no longer needed!
    """
    
    def __init__(
        self,
        claude_adapter: Optional[ClaudeAdapter] = None,
        embedding_dim: int = 768,
        batch_size: int = 16,
        transfer_mode: TransferMode = TransferMode.COMPRESSED,
        verbose: bool = True,
    ):
        """
        Initialize corpus processor.
        
        Args:
            claude_adapter: Claude adapter (or None to create from settings)
            embedding_dim: Embedding dimension
            batch_size: Processing batch size
            transfer_mode: AI2AI transfer mode
            verbose: Show progress
        """
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.transfer_mode = transfer_mode
        self.verbose = verbose
        
        # Create Claude adapter if not provided
        if claude_adapter is None:
            settings = get_settings()
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
            self.claude_adapter = ClaudeAdapter(
                api_key=settings.anthropic_api_key.get_secret_value(),
                embedding_dim=embedding_dim,
                model=settings.default_llm_model,
            )
        else:
            self.claude_adapter = claude_adapter
        
        self.encoder = AI2AIEncoder()
        
        # Statistics
        self.stats = {
            "texts_processed": 0,
            "embeddings_generated": 0,
            "total_tokens": 0,
            "total_bytes": 0,
            "processing_time": 0.0,
        }
    
    def process_texts(
        self,
        texts: List[str],
        output_file: Optional[Path] = None,
        save_format: str = "ai2ai",
    ) -> List[AI2AIMessage]:
        """
        Process list of texts into AI2AI embeddings.
        
        Args:
            texts: List of text strings
            output_file: Optional file to save embeddings
            save_format: "ai2ai" (binary) or "torch" (tensors)
            
        Returns:
            List of AI2AI messages
        """
        if self.verbose:
            print(f"Processing {len(texts)} texts in batches of {self.batch_size}...")
        
        messages = []
        start_time = time.time()
        
        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            disable=not self.verbose,
            desc="Processing batches"
        ):
            batch = texts[i:i + self.batch_size]
            batch_messages = self.claude_adapter.process_corpus(
                batch,
                chunk_size=512,
                batch_size=len(batch)
            )
            messages.extend(batch_messages)
            
            # Update stats
            self.stats["texts_processed"] += len(batch)
            self.stats["embeddings_generated"] += len(batch_messages)
        
        self.stats["processing_time"] = time.time() - start_time
        
        # Save if requested
        if output_file:
            self.save_embeddings(messages, output_file, save_format)
        
        if self.verbose:
            self._print_stats()
        
        return messages
    
    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        save_format: str = "ai2ai",
        chunk_lines: int = 1000,
    ) -> int:
        """
        Process text file into embeddings.
        
        Args:
            input_file: Input text file (one sample per line)
            output_file: Output file for embeddings
            save_format: "ai2ai" or "torch"
            chunk_lines: Process file in chunks
            
        Returns:
            Number of embeddings generated
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if self.verbose:
            print(f"Processing file: {input_file}")
        
        # Read file in chunks
        all_messages = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                line = line.strip()
                if line:
                    chunk.append(line)
                    
                    if len(chunk) >= chunk_lines:
                        messages = self.process_texts(chunk, save_format=save_format)
                        all_messages.extend(messages)
                        chunk = []
            
            # Process remaining
            if chunk:
                messages = self.process_texts(chunk, save_format=save_format)
                all_messages.extend(messages)
        
        # Save all embeddings
        self.save_embeddings(all_messages, output_file, save_format)
        
        return len(all_messages)
    
    def extract_domain_knowledge(
        self,
        domain: str,
        concepts: List[str],
        examples: Optional[List[str]] = None,
        output_file: Optional[Path] = None,
    ) -> KnowledgeTransfer:
        """
        Extract structured knowledge from Claude for specific domain.
        
        This is for targeted knowledge transfer (e.g., "teach NOVA physics").
        
        Args:
            domain: Knowledge domain (e.g., "physics", "math", "rust")
            concepts: List of concepts to extract
            examples: Optional example contexts
            output_file: Optional file to save knowledge
            
        Returns:
            KnowledgeTransfer object
        """
        if self.verbose:
            print(f"\nExtracting knowledge from '{domain}' domain...")
            print(f"Concepts: {', '.join(concepts[:5])}{'...' if len(concepts) > 5 else ''}")
        
        start_time = time.time()
        
        # Use Claude adapter to extract knowledge
        knowledge = self.claude_adapter.extract_knowledge(
            domain=domain,
            concepts=concepts,
            examples=examples
        )
        
        extraction_time = time.time() - start_time
        
        if self.verbose:
            print(f"✓ Extracted {len(concepts)} concepts in {extraction_time:.1f}s")
            print(f"  Confidence: {knowledge.confidence}")
        
        # Save if requested
        if output_file:
            message = knowledge.to_ai2ai_message()
            self.save_embeddings([message], output_file, "ai2ai")
        
        return knowledge
    
    def save_embeddings(
        self,
        messages: List[AI2AIMessage],
        output_file: Path,
        save_format: str = "ai2ai",
    ):
        """
        Save embeddings to file.
        
        Args:
            messages: AI2AI messages
            output_file: Output file path
            save_format: "ai2ai" (binary) or "torch" (tensors)
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == "ai2ai":
            # Save as binary AI2AI format
            with open(output_file, 'wb') as f:
                for message in messages:
                    encoded = self.encoder.encode(message)
                    # Write length prefix
                    f.write(len(encoded).to_bytes(4, byteorder='little'))
                    f.write(encoded)
                    self.stats["total_bytes"] += len(encoded) + 4
        
        elif save_format == "torch":
            # Save as PyTorch tensors
            embeddings = [msg.embeddings for msg in messages]
            torch.save(embeddings, output_file)
            self.stats["total_bytes"] += output_file.stat().st_size
        
        elif save_format == "json":
            # Save metadata as JSON (for inspection)
            data = [msg.to_dict() for msg in messages]
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        if self.verbose:
            print(f"✓ Saved {len(messages)} embeddings to {output_file}")
    
    def load_embeddings(
        self,
        input_file: Path,
        load_format: str = "ai2ai",
    ) -> List[AI2AIMessage]:
        """
        Load embeddings from file.
        
        Args:
            input_file: Input file path
            load_format: "ai2ai" or "torch"
            
        Returns:
            List of AI2AI messages
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if load_format == "ai2ai":
            from ..ai2ai.decoder import AI2AIDecoder
            decoder = AI2AIDecoder()
            
            messages = []
            with open(input_file, 'rb') as f:
                while True:
                    # Read length prefix
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break
                    
                    length = int.from_bytes(length_bytes, byteorder='little')
                    
                    # Read message
                    encoded = f.read(length)
                    message = decoder.decode(encoded)
                    messages.append(message)
            
            return messages
        
        elif load_format == "torch":
            embeddings = torch.load(input_file)
            
            # Convert to AI2AI messages
            messages = []
            for emb in embeddings:
                message = AI2AIMessage(
                    message_type=MessageType.EMBEDDING,
                    embeddings=emb,
                    source_model="loaded",
                    target_model="nova",
                )
                messages.append(message)
            
            return messages
        
        else:
            raise ValueError(f"Unknown load format: {load_format}")
    
    def _print_stats(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("Corpus Processing Statistics")
        print("=" * 50)
        print(f"Texts processed:       {self.stats['texts_processed']}")
        print(f"Embeddings generated:  {self.stats['embeddings_generated']}")
        print(f"Total data:            {self.stats['total_bytes'] / (1024*1024):.2f} MB")
        print(f"Processing time:       {self.stats['processing_time']:.1f}s")
        
        if self.stats['processing_time'] > 0:
            throughput = self.stats['texts_processed'] / self.stats['processing_time']
            print(f"Throughput:            {throughput:.1f} texts/sec")
        
        print("=" * 50)
    
    def stream_process(
        self,
        text_stream: Iterator[str],
        output_file: Path,
        save_format: str = "ai2ai",
    ) -> int:
        """
        Process streaming text data.
        
        Useful for very large corpora that don't fit in memory.
        
        Args:
            text_stream: Iterator yielding text strings
            output_file: Output file
            save_format: Save format
            
        Returns:
            Number of embeddings generated
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open output file
        if save_format == "ai2ai":
            file_handle = open(output_file, 'wb')
        else:
            raise NotImplementedError(f"Streaming not supported for format: {save_format}")
        
        total_processed = 0
        batch = []
        
        try:
            for text in text_stream:
                batch.append(text)
                
                if len(batch) >= self.batch_size:
                    # Process batch
                    messages = self.process_texts(batch, save_format=save_format)
                    
                    # Write to file immediately
                    for message in messages:
                        encoded = self.encoder.encode(message)
                        file_handle.write(len(encoded).to_bytes(4, byteorder='little'))
                        file_handle.write(encoded)
                    
                    total_processed += len(messages)
                    batch = []
            
            # Process remaining
            if batch:
                messages = self.process_texts(batch, save_format=save_format)
                for message in messages:
                    encoded = self.encoder.encode(message)
                    file_handle.write(len(encoded).to_bytes(4, byteorder='little'))
                    file_handle.write(encoded)
                total_processed += len(messages)
        
        finally:
            file_handle.close()
        
        return total_processed
