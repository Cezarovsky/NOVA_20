"""
AI2AI Protocol Demo
Shows direct embedding transfer between AIs (no text overhead).
"""

import torch
import time
from src.ai2ai.protocol import (
    AI2AIMessage, 
    MessageType, 
    TransferMode,
    KnowledgeTransfer,
    ProtocolStats
)
from src.ai2ai.encoder import AI2AIEncoder
from src.ai2ai.decoder import AI2AIDecoder


def demo_basic_transfer():
    """Demo: Basic embedding transfer."""
    print("\n=== Basic Embedding Transfer ===")
    
    # Simulate Claude generating embeddings
    claude_embeddings = torch.randn(128, 768)
    print(f"Claude generates embeddings: {claude_embeddings.shape}")
    
    # Create AI2AI message
    message = AI2AIMessage(
        message_type=MessageType.EMBEDDING,
        embeddings=claude_embeddings,
        source_model="claude-3-haiku",
        target_model="nova",
        transfer_mode=TransferMode.RAW
    )
    
    print(f"Message size: {message.size_bytes() / 1024:.1f} KB")
    
    # Encode (Claude → binary)
    encoder = AI2AIEncoder()
    start = time.time()
    encoded = encoder.encode(message)
    encode_time = (time.time() - start) * 1000
    
    print(f"Encoded in {encode_time:.2f}ms → {len(encoded) / 1024:.1f} KB")
    
    # Decode (binary → NOVA)
    decoder = AI2AIDecoder()
    start = time.time()
    decoded = decoder.decode(encoded)
    decode_time = (time.time() - start) * 1000
    
    print(f"Decoded in {decode_time:.2f}ms")
    print(f"✓ Embeddings match: {torch.allclose(decoded.embeddings, claude_embeddings)}")


def demo_compression():
    """Demo: Compression reduces transfer size."""
    print("\n=== Compression Demo ===")
    
    embeddings = torch.randn(256, 768)
    
    # RAW mode
    raw_msg = AI2AIMessage(
        message_type=MessageType.EMBEDDING,
        embeddings=embeddings.clone(),
        transfer_mode=TransferMode.RAW
    )
    
    # COMPRESSED mode
    compressed_msg = AI2AIMessage(
        message_type=MessageType.EMBEDDING,
        embeddings=embeddings.clone(),
        transfer_mode=TransferMode.COMPRESSED
    )
    
    # QUANTIZED mode  
    quantized_msg = AI2AIMessage(
        message_type=MessageType.EMBEDDING,
        embeddings=embeddings.clone(),
        transfer_mode=TransferMode.QUANTIZED
    )
    
    encoder = AI2AIEncoder(compression_level=9)
    
    raw_size = len(encoder.encode(raw_msg))
    compressed_size = len(encoder.encode(compressed_msg))
    quantized_size = len(encoder.encode(quantized_msg))
    
    print(f"Raw:        {raw_size / 1024:.1f} KB (100%)")
    print(f"Compressed: {compressed_size / 1024:.1f} KB ({compressed_size/raw_size*100:.1f}%)")
    print(f"Quantized:  {quantized_size / 1024:.1f} KB ({quantized_size/raw_size*100:.1f}%)")


def demo_knowledge_transfer():
    """Demo: High-level knowledge transfer."""
    print("\n=== Knowledge Transfer Demo ===")
    
    # Simulate Claude transferring physics concepts to NOVA
    concepts = ["force", "mass", "acceleration", "energy", "momentum"]
    concept_embeddings = torch.randn(len(concepts), 768)
    
    # Relationship graph (adjacency matrix)
    relationships = torch.tensor([
        [1.0, 0.8, 0.9, 0.7, 0.6],  # force relates to...
        [0.8, 1.0, 0.9, 0.5, 0.7],  # mass relates to...
        [0.9, 0.9, 1.0, 0.6, 0.8],  # acceleration relates to...
        [0.7, 0.5, 0.6, 1.0, 0.9],  # energy relates to...
        [0.6, 0.7, 0.8, 0.9, 1.0],  # momentum relates to...
    ])
    
    kt = KnowledgeTransfer(
        concept_embeddings=concept_embeddings,
        concept_names=concepts,
        relationships=relationships,
        domain="physics",
        confidence=0.95
    )
    
    print(f"Transferring {len(concepts)} concepts from 'physics' domain")
    print(f"Concepts: {', '.join(concepts)}")
    print(f"Confidence: {kt.confidence}")
    
    # Convert to AI2AI message
    message = kt.to_ai2ai_message()
    
    encoder = AI2AIEncoder()
    encoded = encoder.encode(message)
    
    print(f"Transfer size: {len(encoded) / 1024:.1f} KB")
    
    # Decode and restore
    decoder = AI2AIDecoder()
    decoded = decoder.decode(encoded)
    restored_kt = KnowledgeTransfer.from_ai2ai_message(decoded)
    
    print(f"✓ Restored {len(restored_kt.concept_names)} concepts")
    print(f"✓ Domain: {restored_kt.domain}")


def demo_attention_transfer():
    """Demo: Transfer attention patterns."""
    print("\n=== Attention Pattern Transfer ===")
    
    seq_len = 64
    num_heads = 8
    
    embeddings = torch.randn(seq_len, 768)
    attention_weights = torch.softmax(torch.randn(num_heads, seq_len, seq_len), dim=-1)
    
    message = AI2AIMessage(
        message_type=MessageType.ATTENTION,
        embeddings=embeddings,
        attention_weights=attention_weights,
        source_model="claude",
        target_model="nova"
    )
    
    print(f"Embeddings: {embeddings.shape}")
    print(f"Attention: {attention_weights.shape}")
    print(f"Total size: {message.size_bytes() / 1024:.1f} KB")
    
    # Encode/decode
    encoder = AI2AIEncoder()
    decoder = AI2AIDecoder()
    
    encoded = encoder.encode(message)
    decoded = decoder.decode(encoded)
    
    print(f"✓ Embeddings match: {torch.allclose(decoded.embeddings, embeddings)}")
    print(f"✓ Attention match: {torch.allclose(decoded.attention_weights, attention_weights)}")


def demo_performance_stats():
    """Demo: Track transfer performance."""
    print("\n=== Performance Statistics ===")
    
    stats = ProtocolStats()
    
    # Simulate multiple transfers
    for i in range(10):
        embeddings = torch.randn(128, 768)
        message = AI2AIMessage(
            message_type=MessageType.EMBEDDING,
            embeddings=embeddings,
            transfer_mode=TransferMode.COMPRESSED
        )
        
        encoder = AI2AIEncoder()
        start = time.time()
        encoded = encoder.encode(message)
        duration = (time.time() - start) * 1000
        
        stats.record_send(message, duration)
    
    print(f"Messages sent: {stats.messages_sent}")
    print(f"Total data: {stats.total_mb_transferred():.2f} MB")
    print(f"Avg transfer time: {stats.avg_transfer_time():.2f}ms")
    print(f"Throughput: {stats.throughput_mbps():.2f} MB/s")


def main():
    """Run all demos."""
    print("=" * 60)
    print("AI2AI Protocol Demo")
    print("Direct embedding transfer (no text overhead)")
    print("=" * 60)
    
    demo_basic_transfer()
    demo_compression()
    demo_knowledge_transfer()
    demo_attention_transfer()
    demo_performance_stats()
    
    print("\n" + "=" * 60)
    print("✓ All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
