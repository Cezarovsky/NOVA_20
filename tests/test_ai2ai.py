"""
Tests for AI2AI protocol: encoding, decoding, compression, transfer.
"""

import pytest
import torch
import numpy as np
from datetime import datetime

from src.ai2ai.protocol import (
    AI2AIMessage, 
    MessageType, 
    TransferMode,
    KnowledgeTransfer,
    ProtocolStats
)
from src.ai2ai.encoder import AI2AIEncoder
from src.ai2ai.decoder import AI2AIDecoder


class TestAI2AIProtocol:
    """Test AI2AI protocol messages."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        embeddings = torch.randn(128, 768)  # [seq_len, emb_dim]
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            source_model="claude-3-haiku",
            target_model="nova"
        )
        
        assert message.message_type == MessageType.EMBEDDING
        assert message.embeddings.shape == (128, 768)
        assert message.embedding_dim == 768
        assert message.sequence_length == 128
        assert message.source_model == "claude-3-haiku"
    
    def test_message_with_attention(self):
        """Test message with attention weights."""
        embeddings = torch.randn(64, 768)
        attention = torch.randn(8, 64, 64)  # [num_heads, seq, seq]
        
        message = AI2AIMessage(
            message_type=MessageType.ATTENTION,
            embeddings=embeddings,
            attention_weights=attention,
            source_model="claude",
            target_model="nova"
        )
        
        assert message.attention_weights is not None
        assert message.attention_weights.shape == (8, 64, 64)
    
    def test_message_size_calculation(self):
        """Test message size calculation."""
        embeddings = torch.randn(100, 768)  # 100 * 768 * 4 bytes = 307,200 bytes
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings
        )
        
        size = message.size_bytes()
        expected = 100 * 768 * 4  # float32 = 4 bytes
        assert size == expected
    
    def test_compression(self):
        """Test message compression."""
        embeddings = torch.randn(128, 768)
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.RAW
        )
        
        compressed = message.compress()
        assert compressed.transfer_mode == TransferMode.COMPRESSED
    
    def test_quantization(self):
        """Test message quantization."""
        embeddings = torch.randn(128, 768)
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.RAW
        )
        
        quantized = message.quantize()
        assert quantized.transfer_mode == TransferMode.QUANTIZED
        assert quantized.embeddings.dtype == torch.int8


class TestAI2AIEncoder:
    """Test AI2AI encoder."""
    
    def test_basic_encoding(self):
        """Test basic message encoding."""
        embeddings = torch.randn(64, 768)
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            source_model="claude",
            target_model="nova"
        )
        
        encoder = AI2AIEncoder()
        encoded = encoder.encode(message)
        
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        assert encoded[:5] == b"AI2AI"  # Magic number
    
    def test_encoding_with_attention(self):
        """Test encoding with attention weights."""
        embeddings = torch.randn(32, 768)
        attention = torch.randn(8, 32, 32)
        
        message = AI2AIMessage(
            message_type=MessageType.ATTENTION,
            embeddings=embeddings,
            attention_weights=attention
        )
        
        encoder = AI2AIEncoder()
        encoded = encoder.encode(message)
        
        assert len(encoded) > embeddings.nelement() * 4  # Should include attention
    
    def test_compression_reduces_size(self):
        """Test that compression actually reduces size."""
        embeddings = torch.randn(128, 768)
        
        # Raw message
        raw_message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.RAW
        )
        
        # Compressed message
        compressed_message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings.clone(),
            transfer_mode=TransferMode.COMPRESSED
        )
        
        encoder = AI2AIEncoder(compression_level=9)
        raw_size = len(encoder.encode(raw_message))
        compressed_size = len(encoder.encode(compressed_message))
        
        # Compression should reduce size (typically 50-70%)
        assert compressed_size < raw_size
        print(f"Compression ratio: {compressed_size / raw_size:.2%}")
    
    def test_size_estimation(self):
        """Test size estimation."""
        embeddings = torch.randn(100, 768)
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings
        )
        
        encoder = AI2AIEncoder()
        estimated = encoder.estimate_size(message)
        actual = len(encoder.encode(message))
        
        # Estimate should be within 20% of actual
        ratio = abs(estimated - actual) / actual
        assert ratio < 0.2, f"Estimation off by {ratio:.1%}"


class TestAI2AIDecoder:
    """Test AI2AI decoder."""
    
    def test_basic_decoding(self):
        """Test basic message decoding."""
        embeddings = torch.randn(64, 768)
        
        original = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            source_model="claude",
            target_model="nova"
        )
        
        # Encode then decode
        encoder = AI2AIEncoder()
        decoder = AI2AIDecoder()
        
        encoded = encoder.encode(original)
        decoded = decoder.decode(encoded)
        
        assert decoded.message_type == original.message_type
        assert decoded.source_model == original.source_model
        assert decoded.target_model == original.target_model
        assert decoded.embeddings.shape == original.embeddings.shape
        
        # Check embedding values match (within float precision)
        assert torch.allclose(decoded.embeddings, original.embeddings, atol=1e-6)
    
    def test_decoding_with_attention(self):
        """Test decoding with attention weights."""
        embeddings = torch.randn(32, 768)
        attention = torch.randn(8, 32, 32)
        
        original = AI2AIMessage(
            message_type=MessageType.ATTENTION,
            embeddings=embeddings,
            attention_weights=attention
        )
        
        encoder = AI2AIEncoder()
        decoder = AI2AIDecoder()
        
        encoded = encoder.encode(original)
        decoded = decoder.decode(encoded)
        
        assert decoded.attention_weights is not None
        assert decoded.attention_weights.shape == attention.shape
        assert torch.allclose(decoded.attention_weights, attention, atol=1e-6)
    
    def test_compressed_roundtrip(self):
        """Test encoding/decoding with compression."""
        embeddings = torch.randn(128, 768)
        
        original = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.COMPRESSED
        )
        
        encoder = AI2AIEncoder(compression_level=6)
        decoder = AI2AIDecoder()
        
        encoded = encoder.encode(original)
        decoded = decoder.decode(encoded)
        
        # Should match exactly after compression roundtrip
        assert torch.allclose(decoded.embeddings, original.embeddings, atol=1e-6)
    
    def test_quantized_roundtrip(self):
        """Test encoding/decoding with quantization."""
        embeddings = torch.randn(64, 768)
        
        original = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.QUANTIZED
        )
        
        encoder = AI2AIEncoder()
        decoder = AI2AIDecoder()
        
        encoded = encoder.encode(original)
        decoded = decoder.decode(encoded)
        
        # Quantization is lossy, so use larger tolerance
        # int8 quantization can have ~1% error
        assert torch.allclose(decoded.embeddings, original.embeddings, atol=0.05)
        
        # But should be much smaller
        raw_message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings.clone(),
            transfer_mode=TransferMode.RAW
        )
        raw_size = len(encoder.encode(raw_message))
        quantized_size = len(encoded)
        
        assert quantized_size < raw_size * 0.3  # Should be <30% of raw
        print(f"Quantization size: {quantized_size / raw_size:.1%} of raw")
    
    def test_validation(self):
        """Test message validation."""
        embeddings = torch.randn(32, 768)
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings
        )
        
        encoder = AI2AIEncoder()
        decoder = AI2AIDecoder()
        
        encoded = encoder.encode(message)
        
        # Valid message
        assert decoder.validate(encoded) is True
        
        # Invalid messages
        assert decoder.validate(b"invalid") is False
        assert decoder.validate(encoded[:10]) is False  # Truncated


class TestKnowledgeTransfer:
    """Test knowledge transfer objects."""
    
    def test_knowledge_transfer_creation(self):
        """Test creating knowledge transfer."""
        concepts = torch.randn(10, 768)  # 10 concepts
        names = [f"concept_{i}" for i in range(10)]
        
        kt = KnowledgeTransfer(
            concept_embeddings=concepts,
            concept_names=names,
            domain="physics"
        )
        
        assert kt.concept_embeddings.shape == (10, 768)
        assert len(kt.concept_names) == 10
        assert kt.domain == "physics"
    
    def test_knowledge_transfer_with_relationships(self):
        """Test knowledge transfer with relationship graph."""
        concepts = torch.randn(5, 768)
        names = ["force", "mass", "acceleration", "energy", "momentum"]
        relationships = torch.rand(5, 5)  # Adjacency matrix
        
        kt = KnowledgeTransfer(
            concept_embeddings=concepts,
            concept_names=names,
            relationships=relationships,
            domain="physics"
        )
        
        assert kt.relationships is not None
        assert kt.relationships.shape == (5, 5)
    
    def test_knowledge_transfer_to_ai2ai(self):
        """Test converting knowledge transfer to AI2AI message."""
        concepts = torch.randn(8, 768)
        names = [f"concept_{i}" for i in range(8)]
        
        kt = KnowledgeTransfer(
            concept_embeddings=concepts,
            concept_names=names,
            domain="math"
        )
        
        message = kt.to_ai2ai_message()
        
        assert isinstance(message, AI2AIMessage)
        assert message.message_type == MessageType.KNOWLEDGE
        assert message.embeddings.shape == (8, 768)
        assert message.metadata["domain"] == "math"
        assert message.metadata["num_concepts"] == 8
    
    def test_knowledge_transfer_roundtrip(self):
        """Test converting to/from AI2AI message."""
        concepts = torch.randn(6, 768)
        names = ["a", "b", "c", "d", "e", "f"]
        relationships = torch.rand(6, 6)
        
        original = KnowledgeTransfer(
            concept_embeddings=concepts,
            concept_names=names,
            relationships=relationships,
            domain="test",
            confidence=0.95
        )
        
        # Convert to message and back
        message = original.to_ai2ai_message()
        restored = KnowledgeTransfer.from_ai2ai_message(message)
        
        assert restored.domain == original.domain
        assert restored.confidence == original.confidence
        assert len(restored.concept_names) == len(original.concept_names)
        assert torch.allclose(restored.concept_embeddings, original.concept_embeddings)


class TestProtocolStats:
    """Test protocol statistics tracking."""
    
    def test_stats_tracking(self):
        """Test basic stats tracking."""
        stats = ProtocolStats()
        
        embeddings = torch.randn(64, 768)
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings
        )
        
        stats.record_send(message, duration_ms=5.2)
        stats.record_send(message, duration_ms=4.8)
        stats.record_receive(message)
        
        assert stats.messages_sent == 2
        assert stats.messages_received == 1
        assert stats.avg_transfer_time() > 0
        assert stats.total_mb_transferred() > 0
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        stats = ProtocolStats()
        
        # Simulate multiple transfers
        for _ in range(10):
            embeddings = torch.randn(128, 768)
            message = AI2AIMessage(
                message_type=MessageType.EMBEDDING,
                embeddings=embeddings
            )
            stats.record_send(message, duration_ms=5.0)
        
        throughput = stats.throughput_mbps()
        assert throughput > 0
        print(f"Throughput: {throughput:.2f} MB/s")


@pytest.mark.integration
class TestAI2AIIntegration:
    """Integration tests for full AI2AI pipeline."""
    
    def test_full_pipeline(self):
        """Test complete encode→decode pipeline."""
        # Create complex message
        embeddings = torch.randn(256, 768)
        attention = torch.randn(12, 256, 256)
        mask = torch.ones(256, 256, dtype=torch.bool)
        
        original = AI2AIMessage(
            message_type=MessageType.ATTENTION,
            embeddings=embeddings,
            attention_weights=attention,
            attention_mask=mask,
            source_model="claude-3-haiku-20240307",
            target_model="nova",
            metadata={
                "domain": "physics",
                "concepts": ["quantum", "mechanics"],
            }
        )
        
        # Encode
        encoder = AI2AIEncoder(compression_level=6)
        encoded = encoder.encode(original)
        
        # Decode
        decoder = AI2AIDecoder()
        decoded = decoder.decode(encoded)
        
        # Verify everything matches
        assert decoded.message_type == original.message_type
        assert decoded.source_model == original.source_model
        assert decoded.target_model == original.target_model
        assert torch.allclose(decoded.embeddings, original.embeddings, atol=1e-6)
        assert torch.allclose(decoded.attention_weights, original.attention_weights, atol=1e-6)
        assert decoded.metadata["domain"] == "physics"
    
    def test_performance_benchmark(self):
        """Benchmark encoding/decoding performance."""
        import time
        
        embeddings = torch.randn(512, 768)  # Large message
        
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.COMPRESSED
        )
        
        encoder = AI2AIEncoder()
        decoder = AI2AIDecoder()
        
        # Benchmark encoding
        start = time.time()
        encoded = encoder.encode(message)
        encode_time = (time.time() - start) * 1000
        
        # Benchmark decoding
        start = time.time()
        decoded = decoder.decode(encoded)
        decode_time = (time.time() - start) * 1000
        
        print(f"\nPerformance (512x768 embeddings):")
        print(f"  Encode: {encode_time:.2f}ms")
        print(f"  Decode: {decode_time:.2f}ms")
        print(f"  Total: {encode_time + decode_time:.2f}ms")
        print(f"  Size: {len(encoded) / 1024:.1f} KB")
        
        # Should be fast (<10ms target)
        assert encode_time < 50, f"Encoding too slow: {encode_time}ms"
        assert decode_time < 50, f"Decoding too slow: {decode_time}ms"
