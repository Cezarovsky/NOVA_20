"""
AI2AI Encoder: Serialize embeddings/tensors for efficient transfer.

Handles compression, quantization, and binary serialization.
"""

import io
import struct
from typing import Optional, BinaryIO
import torch
import numpy as np
import zlib

from .protocol import AI2AIMessage, MessageType, TransferMode


class AI2AIEncoder:
    """
    Encode AI2AI messages to binary format.
    
    Format:
    - Header (64 bytes): version, type, mode, shape, dtype, checksum
    - Embeddings (variable): raw tensor bytes
    - Attention (optional variable): attention tensor bytes
    - Gradients (optional variable): gradient tensor bytes
    - Metadata (variable): JSON metadata
    """
    
    MAGIC_NUMBER = b"AI2AI"
    HEADER_SIZE = 256  # Fixed header size for fast parsing
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize encoder.
        
        Args:
            compression_level: zlib compression level (0-9, 0=none, 9=max)
        """
        self.compression_level = compression_level
    
    def encode(self, message: AI2AIMessage) -> bytes:
        """
        Encode AI2AI message to binary.
        
        Args:
            message: AI2AI message to encode
            
        Returns:
            Binary representation
        """
        buffer = io.BytesIO()
        
        # Write header
        self._write_header(buffer, message)
        
        # Write embeddings
        embeddings_bytes = self._tensor_to_bytes(message.embeddings, message.transfer_mode)
        buffer.write(struct.pack("<I", len(embeddings_bytes)))  # Length prefix
        buffer.write(embeddings_bytes)
        
        # Write attention weights (optional)
        if message.attention_weights is not None:
            attention_bytes = self._tensor_to_bytes(message.attention_weights, message.transfer_mode)
            # Prefix with shape info (4 ints for up to 4D tensor)
            shape = message.attention_weights.shape
            shape_padded = list(shape) + [0] * (4 - len(shape))
            shape_bytes = struct.pack("<IIII", *shape_padded[:4])
            buffer.write(struct.pack("<I", len(attention_bytes) + 16))  # Include shape bytes
            buffer.write(shape_bytes)
            buffer.write(attention_bytes)
        else:
            buffer.write(struct.pack("<I", 0))
        
        # Write attention mask (optional)
        if message.attention_mask is not None:
            mask_bytes = self._tensor_to_bytes(message.attention_mask, message.transfer_mode)
            # Prefix with shape info
            shape = message.attention_mask.shape
            shape_padded = list(shape) + [0] * (4 - len(shape))
            shape_bytes = struct.pack("<IIII", *shape_padded[:4])
            buffer.write(struct.pack("<I", len(mask_bytes) + 16))
            buffer.write(shape_bytes)
            buffer.write(mask_bytes)
        else:
            buffer.write(struct.pack("<I", 0))
        
        # Write gradients (optional)
        if message.gradients is not None:
            grad_bytes = self._tensor_to_bytes(message.gradients, message.transfer_mode)
            # Prefix with shape info
            shape = message.gradients.shape
            shape_padded = list(shape) + [0] * (4 - len(shape))
            shape_bytes = struct.pack("<IIII", *shape_padded[:4])
            buffer.write(struct.pack("<I", len(grad_bytes) + 16))
            buffer.write(shape_bytes)
            buffer.write(grad_bytes)
        else:
            buffer.write(struct.pack("<I", 0))
        
        # Write metadata (JSON compressed)
        import json
        metadata_json = json.dumps(message.to_dict()).encode("utf-8")
        metadata_compressed = zlib.compress(metadata_json, level=self.compression_level)
        buffer.write(struct.pack("<I", len(metadata_compressed)))
        buffer.write(metadata_compressed)
        
        return buffer.getvalue()
    
    def _write_header(self, buffer: BinaryIO, message: AI2AIMessage):
        """Write fixed-size header."""
        header = io.BytesIO()
        
        # Magic number (5 bytes)
        header.write(self.MAGIC_NUMBER)
        
        # Protocol version (3 bytes: major.minor.patch)
        version_parts = message.protocol_version.split(".")
        header.write(struct.pack("<BBB", 
            int(version_parts[0]) if len(version_parts) > 0 else 1,
            int(version_parts[1]) if len(version_parts) > 1 else 0,
            int(version_parts[2]) if len(version_parts) > 2 else 0
        ))
        
        # Message type (1 byte)
        message_type_id = list(MessageType).index(message.message_type)
        header.write(struct.pack("<B", message_type_id))
        
        # Transfer mode (1 byte)
        transfer_mode_id = list(TransferMode).index(message.transfer_mode)
        header.write(struct.pack("<B", transfer_mode_id))
        
        # Shape (4 dimensions, 4 bytes each = 16 bytes)
        shape_padded = list(message.shape) + [0] * (4 - len(message.shape))
        for dim in shape_padded[:4]:
            header.write(struct.pack("<I", dim))
        
        # Embedding dimension (4 bytes)
        header.write(struct.pack("<I", message.embedding_dim))
        
        # Sequence length (4 bytes)
        header.write(struct.pack("<I", message.sequence_length))
        
        # Dtype (16 bytes, null-padded string)
        dtype_bytes = message.dtype.encode("utf-8")[:16].ljust(16, b'\x00')
        header.write(dtype_bytes)
        
        # Source model (32 bytes, null-padded)
        source_bytes = message.source_model.encode("utf-8")[:32].ljust(32, b'\x00')
        header.write(source_bytes)
        
        # Target model (32 bytes, null-padded)
        target_bytes = message.target_model.encode("utf-8")[:32].ljust(32, b'\x00')
        header.write(target_bytes)
        
        # Timestamp (8 bytes, Unix timestamp)
        timestamp_unix = int(message.timestamp.timestamp())
        header.write(struct.pack("<Q", timestamp_unix))
        
        # Flags (1 byte bitfield)
        flags = 0
        if message.attention_weights is not None:
            flags |= 0x01
        if message.attention_mask is not None:
            flags |= 0x02
        if message.gradients is not None:
            flags |= 0x04
        header.write(struct.pack("<B", flags))
        
        # Reserved (pad to 256 bytes)
        header_bytes = header.getvalue()
        padding = b'\x00' * (self.HEADER_SIZE - len(header_bytes))
        buffer.write(header_bytes + padding)
    
    def _tensor_to_bytes(self, tensor: torch.Tensor, mode: TransferMode) -> bytes:
        """Convert tensor to bytes based on transfer mode."""
        if mode == TransferMode.RAW:
            # Raw tensor bytes (fastest)
            return tensor.cpu().numpy().tobytes()
        
        elif mode == TransferMode.COMPRESSED:
            # Compress with zlib
            raw_bytes = tensor.cpu().numpy().tobytes()
            return zlib.compress(raw_bytes, level=self.compression_level)
        
        elif mode == TransferMode.QUANTIZED:
            # Quantize to int8 first
            if tensor.dtype not in [torch.int8, torch.uint8]:
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                scale = 127.0 / max(abs(min_val), abs(max_val))
                quantized = (tensor * scale).round().to(torch.int8)
            else:
                quantized = tensor
                scale = 1.0
            
            # Store scale factor for dequantization (as float32)
            scale_bytes = struct.pack("<f", scale)
            tensor_bytes = quantized.cpu().numpy().tobytes()
            return scale_bytes + tensor_bytes
        
        elif mode == TransferMode.SPARSE:
            # Sparse tensor format (COO)
            if not tensor.is_sparse:
                tensor = tensor.to_sparse()
            
            indices = tensor.indices().cpu().numpy().tobytes()
            values = tensor.values().cpu().numpy().tobytes()
            
            # Format: [indices_len][indices][values_len][values]
            return (struct.pack("<I", len(indices)) + indices + 
                    struct.pack("<I", len(values)) + values)
        
        else:
            raise ValueError(f"Unknown transfer mode: {mode}")
    
    def estimate_size(self, message: AI2AIMessage) -> int:
        """Estimate encoded message size without full encoding."""
        size = self.HEADER_SIZE
        
        # Embeddings
        size += 4 + message.embeddings.element_size() * message.embeddings.nelement()
        
        # Optional tensors
        if message.attention_weights is not None:
            size += 4 + message.attention_weights.element_size() * message.attention_weights.nelement()
        else:
            size += 4
        
        if message.attention_mask is not None:
            size += 4 + message.attention_mask.element_size() * message.attention_mask.nelement()
        else:
            size += 4
        
        if message.gradients is not None:
            size += 4 + message.gradients.element_size() * message.gradients.nelement()
        else:
            size += 4
        
        # Metadata (rough estimate)
        size += 4 + 500  # ~500 bytes for compressed JSON metadata
        
        # Compression reduces size by ~60-70% typically
        if message.transfer_mode == TransferMode.COMPRESSED:
            size = int(size * 0.35)
        elif message.transfer_mode == TransferMode.QUANTIZED:
            size = int(size * 0.25)  # int8 is 4x smaller than float32
        
        return size
