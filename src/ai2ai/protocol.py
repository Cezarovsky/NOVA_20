"""
AI2AI Protocol: Direct embedding/vector communication between AIs.

Eliminates text serialization overhead - transfers raw neural representations.
Inspired by Rust AI2AI implementation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from datetime import datetime


class MessageType(Enum):
    """AI2AI message types."""
    EMBEDDING = "embedding"           # Pure embedding vectors
    ATTENTION = "attention"           # Attention patterns/weights
    GRADIENT = "gradient"             # Training gradients
    KNOWLEDGE = "knowledge"           # Structured knowledge transfer
    CONTEXT = "context"               # Contextual embeddings
    QUERY = "query"                   # Query embeddings
    RESPONSE = "response"             # Response embeddings
    METADATA = "metadata"             # Auxiliary information


class TransferMode(Enum):
    """Transfer optimization modes."""
    RAW = "raw"                       # Raw tensors (largest, fastest)
    COMPRESSED = "compressed"         # Compressed tensors
    QUANTIZED = "quantized"           # 8-bit quantization
    SPARSE = "sparse"                 # Sparse tensor format


@dataclass
class AI2AIMessage:
    """
    Binary message for direct AI-to-AI communication.
    
    No text serialization - just raw neural representations.
    """
    
    # Core data
    message_type: MessageType
    embeddings: torch.Tensor                    # Main embedding tensor
    
    # Optional auxiliary data
    attention_weights: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    
    # Transfer metadata
    transfer_mode: TransferMode = TransferMode.RAW
    shape: tuple = field(default_factory=tuple)
    dtype: str = "float32"
    
    # Semantic metadata
    source_model: str = ""                      # e.g., "claude-3-haiku"
    target_model: str = "nova"
    sequence_length: int = 0
    embedding_dim: int = 768
    
    # Versioning and validation
    protocol_version: str = "1.0"
    checksum: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and populate metadata."""
        if isinstance(self.embeddings, torch.Tensor):
            self.shape = tuple(self.embeddings.shape)
            self.dtype = str(self.embeddings.dtype).replace("torch.", "")
            
            if self.embedding_dim == 768 and len(self.shape) >= 1:
                self.embedding_dim = self.shape[-1]
            
            if self.sequence_length == 0 and len(self.shape) >= 2:
                self.sequence_length = self.shape[-2]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for logging/debugging)."""
        return {
            "message_type": self.message_type.value,
            "transfer_mode": self.transfer_mode.value,
            "shape": self.shape,
            "dtype": self.dtype,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp.isoformat(),
            "has_attention": self.attention_weights is not None,
            "has_gradients": self.gradients is not None,
            "metadata": self.metadata,
        }
    
    def size_bytes(self) -> int:
        """Calculate message size in bytes."""
        size = 0
        
        if isinstance(self.embeddings, torch.Tensor):
            size += self.embeddings.element_size() * self.embeddings.nelement()
        
        if self.attention_weights is not None:
            size += self.attention_weights.element_size() * self.attention_weights.nelement()
        
        if self.attention_mask is not None:
            size += self.attention_mask.element_size() * self.attention_mask.nelement()
            
        if self.gradients is not None:
            size += self.gradients.element_size() * self.gradients.nelement()
        
        return size
    
    def compress(self) -> 'AI2AIMessage':
        """Compress embeddings (in-place modification)."""
        if self.transfer_mode == TransferMode.RAW:
            # Apply compression
            self.embeddings = self._compress_tensor(self.embeddings)
            if self.attention_weights is not None:
                self.attention_weights = self._compress_tensor(self.attention_weights)
            self.transfer_mode = TransferMode.COMPRESSED
        return self
    
    def quantize(self) -> 'AI2AIMessage':
        """Quantize to 8-bit (in-place modification)."""
        if self.transfer_mode == TransferMode.RAW:
            self.embeddings = self._quantize_tensor(self.embeddings)
            if self.attention_weights is not None:
                self.attention_weights = self._quantize_tensor(self.attention_weights)
            self.transfer_mode = TransferMode.QUANTIZED
        return self
    
    @staticmethod
    def _compress_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Apply lossless compression to tensor."""
        # TODO: Implement proper compression (e.g., zlib on serialized tensor)
        # For now, just return as-is
        return tensor
    
    @staticmethod
    def _quantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to int8."""
        # Scale to [-127, 127] range
        min_val = tensor.min()
        max_val = tensor.max()
        scale = 127.0 / max(abs(min_val), abs(max_val))
        
        quantized = (tensor * scale).round().to(torch.int8)
        return quantized


@dataclass 
class KnowledgeTransfer:
    """
    High-level knowledge transfer between AIs.
    
    Used during training: Claude → NOVA knowledge injection.
    """
    
    # Concept embeddings
    concept_embeddings: torch.Tensor            # [num_concepts, embedding_dim]
    concept_names: List[str]                    # Human-readable names
    
    # Relationship graph
    relationships: Optional[torch.Tensor] = None  # [num_concepts, num_concepts] adjacency
    relationship_types: Optional[List[str]] = None
    
    # Contextual examples
    example_embeddings: Optional[torch.Tensor] = None
    example_contexts: Optional[List[str]] = None
    
    # Domain metadata
    domain: str = "general"                     # "physics", "math", "rust", etc.
    confidence: float = 1.0                     # Transfer confidence
    
    # Training hints
    learning_rate_hint: Optional[float] = None
    epochs_hint: Optional[int] = None
    
    def to_ai2ai_message(self) -> AI2AIMessage:
        """Convert to AI2AI protocol message."""
        return AI2AIMessage(
            message_type=MessageType.KNOWLEDGE,
            embeddings=self.concept_embeddings,
            attention_weights=self.relationships,
            source_model="claude",
            target_model="nova",
            metadata={
                "domain": self.domain,
                "confidence": self.confidence,
                "num_concepts": len(self.concept_names),
                "concept_names": self.concept_names,
                "relationship_types": self.relationship_types,
                "learning_rate_hint": self.learning_rate_hint,
                "epochs_hint": self.epochs_hint,
            }
        )
    
    @classmethod
    def from_ai2ai_message(cls, message: AI2AIMessage) -> 'KnowledgeTransfer':
        """Create from AI2AI protocol message."""
        return cls(
            concept_embeddings=message.embeddings,
            concept_names=message.metadata.get("concept_names", []),
            relationships=message.attention_weights,
            relationship_types=message.metadata.get("relationship_types"),
            domain=message.metadata.get("domain", "general"),
            confidence=message.metadata.get("confidence", 1.0),
            learning_rate_hint=message.metadata.get("learning_rate_hint"),
            epochs_hint=message.metadata.get("epochs_hint"),
        )


class ProtocolStats:
    """Track AI2AI protocol performance."""
    
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.transfer_times: List[float] = []
        self.compression_ratios: List[float] = []
    
    def record_send(self, message: AI2AIMessage, duration_ms: float):
        """Record sent message."""
        self.messages_sent += 1
        size = message.size_bytes()
        self.total_bytes_sent += size
        self.transfer_times.append(duration_ms)
    
    def record_receive(self, message: AI2AIMessage):
        """Record received message."""
        self.messages_received += 1
        self.total_bytes_received += message.size_bytes()
    
    def avg_transfer_time(self) -> float:
        """Average transfer time in ms."""
        return np.mean(self.transfer_times) if self.transfer_times else 0.0
    
    def total_mb_transferred(self) -> float:
        """Total data transferred in MB."""
        return (self.total_bytes_sent + self.total_bytes_received) / (1024 * 1024)
    
    def throughput_mbps(self) -> float:
        """Throughput in MB/s."""
        if not self.transfer_times:
            return 0.0
        total_time_s = sum(self.transfer_times) / 1000.0
        return self.total_mb_transferred() / total_time_s if total_time_s > 0 else 0.0
