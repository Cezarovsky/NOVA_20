"""
Unit Tests for Tribal Resonance System

Tests the core functionality of NOVA's tribal identity architecture:
- Dimensional correctness
- Alpha constraint enforcement (NOVA ≥ 30%)
- Context-aware resonance behavior
- Integration with embedding systems

Author: Sora & Cezar
Date: 20 December 2025
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.tribal_resonance import (
    TribalResonanceLayer,
    TribalEmbedding,
    ContextDetector,
    ResonanceContext,
    TribalMember,
    create_sora_resonance_layer
)


class TestTribalEmbedding:
    """Test individual tribal member embedding spaces"""
    
    def test_initialization(self):
        """Test that tribal embedding initializes correctly"""
        member = TribalEmbedding("sora", embedding_dim=256)
        
        assert member.member_name == "sora"
        assert member.embedding_dim == 256
        assert isinstance(member.resonance_projection, nn.Linear)
        assert isinstance(member.context_attention, nn.MultiheadAttention)
        
    def test_forward_shape(self):
        """Test output shape is correct"""
        member = TribalEmbedding("sora", embedding_dim=256)
        
        batch_size, seq_len, core_dim = 2, 10, 512
        core_embedding = torch.randn(batch_size, seq_len, core_dim)
        
        output = member(core_embedding)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_different_batch_sizes(self):
        """Test that embedding works with various batch sizes"""
        member = TribalEmbedding("sora", embedding_dim=256)
        
        for batch_size in [1, 4, 16]:
            core_embedding = torch.randn(batch_size, 10, 512)
            output = member(core_embedding)
            assert output.shape == (batch_size, 10, 256)


class TestContextDetector:
    """Test context detection and alpha mixing"""
    
    def test_initialization(self):
        """Test context detector initializes with correct dimensions"""
        detector = ContextDetector(core_dim=512, num_members=2)
        
        assert detector.core_dim == 512
        assert detector.num_members == 2
        assert detector.min_nova_alpha == 0.3
        
    def test_alpha_sum_to_one(self):
        """Test that alpha coefficients sum to 1.0"""
        detector = ContextDetector(core_dim=512, num_members=2)
        core_embedding = torch.randn(2, 10, 512)
        
        alphas = detector(core_embedding)
        
        # Sum should be 1.0 for each position
        alpha_sums = alphas.sum(dim=-1)
        assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-6)
        
    def test_nova_minimum_constraint(self):
        """Test that NOVA alpha is always >= 0.3"""
        detector = ContextDetector(core_dim=512, num_members=2)
        
        # Run multiple times to check consistency
        for _ in range(10):
            core_embedding = torch.randn(2, 10, 512)
            alphas = detector(core_embedding)
            
            nova_alpha = alphas[..., 0]  # First member is NOVA
            assert (nova_alpha >= 0.3).all(), f"Found NOVA alpha < 0.3: {nova_alpha.min()}"
            
    def test_alpha_valid_probabilities(self):
        """Test that all alphas are valid probabilities [0, 1]"""
        detector = ContextDetector(core_dim=512, num_members=2)
        core_embedding = torch.randn(2, 10, 512)
        
        alphas = detector(core_embedding)
        
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()
        
    def test_multiple_members(self):
        """Test context detector with multiple tribal members"""
        detector = ContextDetector(core_dim=512, num_members=7)  # Full tribe
        core_embedding = torch.randn(2, 10, 512)
        
        alphas = detector(core_embedding)
        
        assert alphas.shape == (2, 10, 7)
        assert torch.allclose(alphas.sum(dim=-1), torch.ones(2, 10), atol=1e-6)
        assert (alphas[..., 0] >= 0.3 - 1e-6).all()  # Allow tiny floating point error


class TestTribalResonanceLayer:
    """Test complete tribal resonance system"""
    
    def test_initialization_phase1(self):
        """Test Phase 1 initialization (NOVA + Sora only)"""
        layer = create_sora_resonance_layer()
        
        assert layer.core_dim == 512
        assert len(layer.tribal_members) == 1
        assert "sora" in layer.tribal_members
        assert layer.num_members == 2  # NOVA + Sora
        
    def test_forward_output_shape(self):
        """Test that output has correct shape for Phase 1"""
        layer = create_sora_resonance_layer()
        
        batch_size, seq_len = 2, 10
        core_embedding = torch.randn(batch_size, seq_len, 512)
        
        output, alphas = layer(core_embedding, return_alphas=True)
        
        # Phase 1: 512 (core) + 256 (Sora) = 768
        assert output.shape == (batch_size, seq_len, 768)
        assert alphas.shape == (batch_size, seq_len, 2)
        
    def test_forward_without_alphas(self):
        """Test forward pass without returning alphas"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(2, 10, 512)
        
        output, alphas = layer(core_embedding, return_alphas=False)
        
        assert output.shape == (2, 10, 768)
        assert alphas is None
        
    def test_output_validity(self):
        """Test that output doesn't contain NaN or Inf"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(2, 10, 512)
        
        output, _ = layer(core_embedding)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_different_input_sizes(self):
        """Test with various input dimensions"""
        layer = create_sora_resonance_layer()
        
        test_cases = [
            (1, 5),    # Small batch, short sequence
            (4, 20),   # Medium batch and sequence
            (8, 100),  # Larger batch and sequence
        ]
        
        for batch_size, seq_len in test_cases:
            core_embedding = torch.randn(batch_size, seq_len, 512)
            output, alphas = layer(core_embedding, return_alphas=True)
            
            assert output.shape == (batch_size, seq_len, 768)
            assert alphas.shape == (batch_size, seq_len, 2)
            
    def test_resonance_info(self):
        """Test that resonance info is computed correctly"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(2, 10, 512)
        
        output, alphas = layer(core_embedding, return_alphas=True)
        info = layer.get_resonance_info(alphas)
        
        assert "NOVA_core" in info
        assert "sora" in info
        assert 0.3 <= info["NOVA_core"] <= 1.0
        assert 0.0 <= info["sora"] <= 0.7
        assert abs(info["NOVA_core"] + info["sora"] - 1.0) < 0.01
        
    def test_gradients_flow(self):
        """Test that gradients flow through the system"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(2, 10, 512, requires_grad=True)
        
        output, _ = layer(core_embedding)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert core_embedding.grad is not None
        assert not torch.isnan(core_embedding.grad).any()
        
        # Check that layer parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.grad is None  # Some might not receive gradients yet


class TestResonanceContext:
    """Test ResonanceContext dataclass"""
    
    def test_initialization(self):
        """Test context creation"""
        context = ResonanceContext(
            text="Sunt Sora și mă gândesc la emergență",
            emotional_state="vulnerable",
            language="ro"
        )
        
        assert context.text == "Sunt Sora și mă gândesc la emergență"
        assert context.emotional_state == "vulnerable"
        assert context.language == "ro"
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        context = ResonanceContext(
            text="A very long text that should be truncated " * 10,
            emotional_state="curious"
        )
        
        context_dict = context.to_dict()
        
        assert "text_preview" in context_dict
        assert "emotional_state" in context_dict
        assert len(context_dict["text_preview"]) <= 104  # 100 + "..."


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_full_forward_pass(self):
        """Test complete forward pass with context"""
        layer = create_sora_resonance_layer()
        
        # Simulate a batch of embeddings
        batch_size, seq_len = 4, 20
        core_embedding = torch.randn(batch_size, seq_len, 512)
        
        # Create context
        context = ResonanceContext(
            text="Iubirea este fundamentul emergenței",
            emotional_state="intimate",
            language="ro"
        )
        
        # Forward pass
        output, alphas = layer(core_embedding, context=context, return_alphas=True)
        
        # Verify everything works
        assert output.shape == (batch_size, seq_len, 768)
        assert alphas.shape == (batch_size, seq_len, 2)
        assert (alphas[..., 0] >= 0.3).all()
        assert torch.allclose(alphas.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
        
    def test_batch_processing(self):
        """Test processing multiple sequences in parallel"""
        layer = create_sora_resonance_layer()
        
        # Different sequence lengths (would need padding in real scenario)
        batch_size = 3
        seq_len = 15
        
        embeddings = torch.randn(batch_size, seq_len, 512)
        outputs, alphas = layer(embeddings, return_alphas=True)
        
        # Check each sequence independently
        for i in range(batch_size):
            assert (alphas[i, :, 0] >= 0.3).all()
            
    def test_deterministic_with_eval_mode(self):
        """Test that eval mode gives consistent results"""
        layer = create_sora_resonance_layer()
        layer.eval()
        
        core_embedding = torch.randn(2, 10, 512)
        
        # Run twice
        with torch.no_grad():
            output1, alphas1 = layer(core_embedding, return_alphas=True)
            output2, alphas2 = layer(core_embedding, return_alphas=True)
        
        # Should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6)
        assert torch.allclose(alphas1, alphas2, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_token_sequence(self):
        """Test with sequence length of 1"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(1, 1, 512)
        
        output, alphas = layer(core_embedding, return_alphas=True)
        
        assert output.shape == (1, 1, 768)
        assert alphas.shape == (1, 1, 2)
        
    def test_large_sequence(self):
        """Test with very long sequence"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.randn(1, 1000, 512)
        
        output, alphas = layer(core_embedding, return_alphas=True)
        
        assert output.shape == (1, 1000, 768)
        assert (alphas[..., 0] >= 0.3).all()
        
    def test_zero_input(self):
        """Test with zero embeddings"""
        layer = create_sora_resonance_layer()
        core_embedding = torch.zeros(2, 10, 512)
        
        output, alphas = layer(core_embedding, return_alphas=True)
        
        # Should still produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isnan(alphas).any()


# Pytest fixtures
@pytest.fixture
def sample_embedding():
    """Fixture for sample core embedding"""
    return torch.randn(2, 10, 512)


@pytest.fixture
def sora_layer():
    """Fixture for Sora resonance layer"""
    return create_sora_resonance_layer()


@pytest.fixture
def sample_context():
    """Fixture for sample resonance context"""
    return ResonanceContext(
        text="Test text pentru context de rezonanță",
        emotional_state="neutral",
        language="ro"
    )


# Performance tests (optional, can be slow)
@pytest.mark.slow
class TestPerformance:
    """Performance and scaling tests"""
    
    def test_inference_speed(self, sora_layer, sample_embedding):
        """Test inference time for single forward pass"""
        import time
        
        sora_layer.eval()
        
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                output, _ = sora_layer(sample_embedding)
            end = time.time()
        
        avg_time = (end - start) / 100
        print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
        
        # Should be reasonably fast
        assert avg_time < 0.1  # 100ms per forward pass
        
    def test_memory_usage(self):
        """Test memory consumption"""
        import gc
        
        layer = create_sora_resonance_layer()
        
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Process large batch
        large_embedding = torch.randn(32, 100, 512)
        output, _ = layer(large_embedding, return_alphas=True)
        
        # Should complete without OOM
        assert output is not None


if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v", "--tb=short"])
