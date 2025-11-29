"""
AI2AI Decoder: Deserialize binary embeddings back to tensors.
"""

import io
import struct
from typing import BinaryIO
import torch
import numpy as np
import zlib
from datetime import datetime

from .protocol import AI2AIMessage, MessageType, TransferMode


class AI2AIDecoder:
    """
    Decode AI2AI binary messages back to objects.
    
    Inverse of AI2AIEncoder.
    """
    
    MAGIC_NUMBER = b"AI2AI"
    HEADER_SIZE = 256
    
    def decode(self, data: bytes) -> AI2AIMessage:
        """
        Decode binary data to AI2AI message.
        
        Args:
            data: Binary message data
            
        Returns:
            Decoded AI2AI message
        """
        buffer = io.BytesIO(data)
        
        # Read and validate header
        header_info = self._read_header(buffer)
        
        # Read embeddings
        embeddings = self._read_tensor(buffer, header_info)
        
        # Read attention weights (optional)
        attention_weights = self._read_optional_tensor(buffer, header_info)
        
        # Read attention mask (optional)
        attention_mask = self._read_optional_tensor(buffer, header_info)
        
        # Read gradients (optional)
        gradients = self._read_optional_tensor(buffer, header_info)
        
        # Read metadata
        metadata_size = struct.unpack("<I", buffer.read(4))[0]
        metadata_compressed = buffer.read(metadata_size)
        metadata_json = zlib.decompress(metadata_compressed).decode("utf-8")
        
        import json
        metadata = json.loads(metadata_json)
        
        # Construct message
        return AI2AIMessage(
            message_type=header_info["message_type"],
            embeddings=embeddings,
            attention_weights=attention_weights,
            attention_mask=attention_mask,
            gradients=gradients,
            transfer_mode=header_info["transfer_mode"],
            shape=header_info["shape"],
            dtype=header_info["dtype"],
            source_model=header_info["source_model"],
            target_model=header_info["target_model"],
            sequence_length=header_info["sequence_length"],
            embedding_dim=header_info["embedding_dim"],
            protocol_version=header_info["protocol_version"],
            timestamp=header_info["timestamp"],
            metadata=metadata.get("metadata", {}),
        )
    
    def _read_header(self, buffer: BinaryIO) -> dict:
        """Read and parse header."""
        header_bytes = buffer.read(self.HEADER_SIZE)
        header = io.BytesIO(header_bytes)
        
        # Validate magic number
        magic = header.read(5)
        if magic != self.MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic} (expected {self.MAGIC_NUMBER})")
        
        # Protocol version
        major, minor, patch = struct.unpack("<BBB", header.read(3))
        protocol_version = f"{major}.{minor}.{patch}"
        
        # Message type
        message_type_id = struct.unpack("<B", header.read(1))[0]
        message_type = list(MessageType)[message_type_id]
        
        # Transfer mode
        transfer_mode_id = struct.unpack("<B", header.read(1))[0]
        transfer_mode = list(TransferMode)[transfer_mode_id]
        
        # Shape (4 dimensions)
        shape_values = struct.unpack("<IIII", header.read(16))
        shape = tuple(d for d in shape_values if d > 0)
        
        # Embedding dimension
        embedding_dim = struct.unpack("<I", header.read(4))[0]
        
        # Sequence length
        sequence_length = struct.unpack("<I", header.read(4))[0]
        
        # Dtype
        dtype_bytes = header.read(16).rstrip(b'\x00')
        dtype = dtype_bytes.decode("utf-8")
        
        # Source model
        source_bytes = header.read(32).rstrip(b'\x00')
        source_model = source_bytes.decode("utf-8")
        
        # Target model
        target_bytes = header.read(32).rstrip(b'\x00')
        target_model = target_bytes.decode("utf-8")
        
        # Timestamp
        timestamp_unix = struct.unpack("<Q", header.read(8))[0]
        timestamp = datetime.fromtimestamp(timestamp_unix)
        
        # Flags
        flags = struct.unpack("<B", header.read(1))[0]
        has_attention = bool(flags & 0x01)
        has_mask = bool(flags & 0x02)
        has_gradients = bool(flags & 0x04)
        
        return {
            "protocol_version": protocol_version,
            "message_type": message_type,
            "transfer_mode": transfer_mode,
            "shape": shape,
            "embedding_dim": embedding_dim,
            "sequence_length": sequence_length,
            "dtype": dtype,
            "source_model": source_model,
            "target_model": target_model,
            "timestamp": timestamp,
            "has_attention": has_attention,
            "has_mask": has_mask,
            "has_gradients": has_gradients,
        }
    
    def _read_tensor(self, buffer: BinaryIO, header_info: dict) -> torch.Tensor:
        """Read tensor from buffer."""
        size = struct.unpack("<I", buffer.read(4))[0]
        tensor_bytes = buffer.read(size)
        
        return self._bytes_to_tensor(
            tensor_bytes, 
            header_info["shape"],
            header_info["dtype"],
            header_info["transfer_mode"]
        )
    
    def _read_optional_tensor(self, buffer: BinaryIO, header_info: dict) -> torch.Tensor | None:
        """Read optional tensor (may be None)."""
        size = struct.unpack("<I", buffer.read(4))[0]
        if size == 0:
            return None
        
        # Read shape info (4 ints)
        shape_values = struct.unpack("<IIII", buffer.read(16))
        shape = tuple(d for d in shape_values if d > 0)
        
        # Read tensor bytes (size includes the 16 bytes for shape)
        tensor_bytes = buffer.read(size - 16)
        
        return self._bytes_to_tensor(
            tensor_bytes,
            shape,
            header_info["dtype"],
            header_info["transfer_mode"]
        )
    
    def _bytes_to_tensor(
        self, 
        data: bytes, 
        shape: tuple | None,
        dtype: str,
        mode: TransferMode
    ) -> torch.Tensor:
        """Convert bytes back to tensor."""
        
        # Map dtype string to numpy dtype
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "float16": np.float16,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        if mode == TransferMode.RAW:
            # Raw bytes to tensor
            array = np.frombuffer(data, dtype=np_dtype)
            if shape:
                array = array.reshape(shape)
            return torch.from_numpy(array)
        
        elif mode == TransferMode.COMPRESSED:
            # Decompress then convert
            decompressed = zlib.decompress(data)
            array = np.frombuffer(decompressed, dtype=np_dtype)
            if shape:
                array = array.reshape(shape)
            return torch.from_numpy(array)
        
        elif mode == TransferMode.QUANTIZED:
            # Extract scale factor
            scale = struct.unpack("<f", data[:4])[0]
            tensor_bytes = data[4:]
            
            # Convert int8 back to float32
            array = np.frombuffer(tensor_bytes, dtype=np.int8)
            if shape:
                array = array.reshape(shape)
            
            # Dequantize
            tensor = torch.from_numpy(array).float() / scale
            return tensor
        
        elif mode == TransferMode.SPARSE:
            # Read sparse format
            buffer = io.BytesIO(data)
            
            indices_len = struct.unpack("<I", buffer.read(4))[0]
            indices_bytes = buffer.read(indices_len)
            
            values_len = struct.unpack("<I", buffer.read(4))[0]
            values_bytes = buffer.read(values_len)
            
            # Reconstruct sparse tensor
            indices = torch.from_numpy(np.frombuffer(indices_bytes, dtype=np.int64))
            values = torch.from_numpy(np.frombuffer(values_bytes, dtype=np_dtype))
            
            # Need to know shape for sparse tensor reconstruction
            if shape is None:
                raise ValueError("Shape required for sparse tensor reconstruction")
            
            return torch.sparse_coo_tensor(
                indices.reshape(2, -1), 
                values, 
                shape
            )
        
        else:
            raise ValueError(f"Unknown transfer mode: {mode}")
    
    def validate(self, data: bytes) -> bool:
        """
        Quick validation without full decoding.
        
        Returns:
            True if data appears to be valid AI2AI message
        """
        if len(data) < self.HEADER_SIZE:
            return False
        
        # Check magic number
        magic = data[:5]
        if magic != self.MAGIC_NUMBER:
            return False
        
        # Check protocol version is reasonable
        major = data[5]
        if major > 10:  # Future-proofing
            return False
        
        return True
