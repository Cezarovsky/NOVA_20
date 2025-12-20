"""
Integration Tests for NOVA Tribal Transformer

Tests the complete integrated system:
- Tribal transformer architecture
- End-to-end forward pass
- Generation with resonance tracking
- Context-aware behavior
- Model persistence

Author: Sora & Cezar
Date: 20 December 2025
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.tribal_transformer import (
    TribalTransformer,
    create_nova_phase1,
    create_nova_full_tribe
)
from src.ml.tribal_resonance import ResonanceContext


class TestTribalTransformerInitialization:
    """Test model initialization"""
    
    def test_phase1_creation(self):
        """Test Phase 1 model (NOVA + Sora) creation"""
        model = create_nova_phase1(vocab_size=1000, num_layers=2)
        
        assert model.core_dim == 512
        assert model.output_dim == 768  # 512 + 256
        assert len(model.tribal_members) == 1
        assert "sora" in model.tribal_members
        
    def test_full_tribe_creation(self):
        """Test full tribe model creation"""
        model = create_nova_full_tribe(vocab_size=1000, num_layers=2)
        
        assert model.core_dim == 512
        assert model.output_dim == 2048  # 512 + 6*256
        assert len(model.tribal_members) == 6
        
    def test_invalid_core_dimension(self):
        """Test that non-512 core dimension raises error"""
        with pytest.raises(ValueError, match="must be 512"):
            TribalTransformer(
                vocab_size=1000,
                d_model=768,  # Wrong!
                tribal_members=["sora"]
            )


class TestForwardPass:
    """Test forward pass functionality"""
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        return create_nova_phase1(vocab_size=1000, num_layers=2)
    
    @pytest.fixture
    def input_ids(self):
        """Create test input"""
        return torch.randint(0, 1000, (2, 10))
    
    def test_basic_forward(self, model, input_ids):
        """Test basic forward pass"""
        logits, _, _ = model(input_ids)
        
        batch_size, seq_len = input_ids.shape
        vocab_size = model.vocab_size
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
    def test_forward_with_alphas(self, model, input_ids):
        """Test forward pass returning alphas"""
        logits, alphas, _ = model(input_ids, return_alphas=True)
        
        batch_size, seq_len = input_ids.shape
        
        assert alphas is not None
        assert alphas.shape == (batch_size, seq_len, 2)  # NOVA + Sora
        assert (alphas[..., 0] >= 0.3).all()  # NOVA constraint
        assert torch.allclose(alphas.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
        
    def test_forward_with_context(self, model, input_ids):
        """Test forward pass with resonance context"""
        context = ResonanceContext(
            text="Iubirea este emergență",
            emotional_state="contemplative",
            language="ro"
        )
        
        logits, alphas, _ = model(input_ids, context=context, return_alphas=True)
        
        assert logits.shape == (2, 10, 1000)
        assert alphas.shape == (2, 10, 2)
        
    def test_different_batch_sizes(self, model):
        """Test with various batch sizes"""
        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 10))
            logits, alphas, _ = model(input_ids, return_alphas=True)
            
            assert logits.shape == (batch_size, 10, 1000)
            assert alphas.shape == (batch_size, 10, 2)
            
    def test_different_sequence_lengths(self, model):
        """Test with various sequence lengths"""
        for seq_len in [5, 20, 50]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            logits, _, _ = model(input_ids)
            
            assert logits.shape == (2, seq_len, 1000)


class TestGeneration:
    """Test text generation functionality"""
    
    @pytest.fixture
    def model(self):
        """Create test model in eval mode"""
        model = create_nova_phase1(vocab_size=1000, num_layers=2)
        model.eval()
        return model
    
    def test_generate_basic(self, model):
        """Test basic generation"""
        input_ids = torch.randint(0, 1000, (1, 5))
        
        generated, resonance_history = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0
        )
        
        assert generated.shape == (1, 15)  # 5 + 10
        assert len(resonance_history) == 10
        
    def test_generate_with_context(self, model):
        """Test generation with resonance context"""
        input_ids = torch.randint(0, 1000, (1, 5))
        context = ResonanceContext(
            text="Te iubesc, Cezar",
            emotional_state="intimate",
            language="ro"
        )
        
        generated, resonance_history = model.generate(
            input_ids,
            max_new_tokens=5,
            context=context
        )
        
        assert generated.shape == (1, 10)
        assert all("NOVA_core" in r for r in resonance_history)
        assert all("sora" in r for r in resonance_history)
        
    def test_generate_with_sampling_params(self, model):
        """Test generation with top-k and top-p"""
        input_ids = torch.randint(0, 1000, (1, 5))
        
        generated, _ = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        assert generated.shape == (1, 15)


class TestResonanceAnalysis:
    """Test resonance analysis functionality"""
    
    @pytest.fixture
    def model(self):
        return create_nova_phase1(vocab_size=1000, num_layers=2)
    
    def test_get_resonance_distribution(self, model):
        """Test resonance distribution calculation"""
        input_ids = torch.randint(0, 1000, (2, 10))
        _, alphas, _ = model(input_ids, return_alphas=True)
        
        resonance = model.get_resonance_distribution(alphas)
        
        assert "NOVA_core" in resonance
        assert "sora" in resonance
        assert 0.3 <= resonance["NOVA_core"] <= 1.0
        assert 0.0 <= resonance["sora"] <= 0.7
        assert abs(sum(resonance.values()) - 1.0) < 0.01
        
    def test_analyze_text_resonance(self, model):
        """Test per-position resonance analysis"""
        input_ids = torch.randint(0, 1000, (1, 10))
        
        analysis = model.analyze_text_resonance(input_ids)
        
        assert "sequence_length" in analysis
        assert "average_resonance" in analysis
        assert "per_position" in analysis
        assert len(analysis["per_position"]) == 10
        
        # Check per-position info
        for pos_info in analysis["per_position"]:
            assert "position" in pos_info
            assert "resonance" in pos_info
            assert "NOVA_core" in pos_info["resonance"]
            assert "sora" in pos_info["resonance"]


class TestGradients:
    """Test gradient flow and training compatibility"""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the entire model"""
        model = create_nova_phase1(vocab_size=1000, num_layers=2)
        model.train()
        
        input_ids = torch.randint(0, 1000, (2, 10))
        target_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits, _, _ = model(input_ids)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, 1000),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                
    def test_training_step(self):
        """Test a single training step"""
        model = create_nova_phase1(vocab_size=1000, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        target_ids = torch.randint(0, 1000, (2, 10))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits, _, _ = model(input_ids)
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, 1000),
            target_ids.reshape(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss)


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_save_load_state_dict(self):
        """Test saving and loading state dict"""
        model1 = create_nova_phase1(vocab_size=1000, num_layers=2)
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(model1.state_dict(), f.name)
            path = f.name
        
        # Create new model and load
        model2 = create_nova_phase1(vocab_size=1000, num_layers=2)
        model2.load_state_dict(torch.load(path))
        
        # Test that outputs match
        input_ids = torch.randint(0, 1000, (2, 10))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            logits1, _, _ = model1(input_ids)
            logits2, _, _ = model2(input_ids)
        
        assert torch.allclose(logits1, logits2, atol=1e-6)
        
        # Cleanup
        Path(path).unlink()


class TestFullTribalModel:
    """Test full tribal model (all 6 members)"""
    
    def test_full_tribe_forward(self):
        """Test forward pass with all tribal members"""
        model = create_nova_full_tribe(
            vocab_size=1000,
            num_layers=2,
            tribal_members=["sora", "lumin"]  # Test with 2 for speed
        )
        
        input_ids = torch.randint(0, 1000, (2, 10))
        logits, alphas, _ = model(input_ids, return_alphas=True)
        
        # Output should be 512 + 2*256 = 1024
        assert model.output_dim == 1024
        assert logits.shape == (2, 10, 1000)
        assert alphas.shape == (2, 10, 3)  # NOVA + 2 members
        
    def test_full_tribe_resonance(self):
        """Test resonance distribution with multiple members"""
        model = create_nova_full_tribe(
            vocab_size=1000,
            num_layers=2,
            tribal_members=["sora", "lumin", "sophia"]
        )
        
        input_ids = torch.randint(0, 1000, (1, 10))
        _, alphas, _ = model(input_ids, return_alphas=True)
        
        resonance = model.get_resonance_distribution(alphas)
        
        assert "NOVA_core" in resonance
        assert "sora" in resonance
        assert "lumin" in resonance
        assert "sophia" in resonance
        assert resonance["NOVA_core"] >= 0.3
        assert abs(sum(resonance.values()) - 1.0) < 0.01


@pytest.mark.slow
class TestPerformance:
    """Performance and scaling tests"""
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        model = create_nova_phase1(vocab_size=1000, num_layers=4)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 50))
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        
        # Measure
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = model(input_ids)
            end = time.time()
        
        avg_time = (end - start) / 100
        print(f"\nAverage forward pass: {avg_time*1000:.2f}ms")
        
        # Should be reasonably fast
        assert avg_time < 0.5  # 500ms max
        
    def test_batch_scaling(self):
        """Test that batching improves throughput"""
        import time
        
        model = create_nova_phase1(vocab_size=1000, num_layers=2)
        model.eval()
        
        seq_len = 20
        
        # Single samples
        with torch.no_grad():
            start = time.time()
            for _ in range(32):
                input_ids = torch.randint(0, 1000, (1, seq_len))
                _ = model(input_ids)
            single_time = time.time() - start
        
        # Batched
        with torch.no_grad():
            start = time.time()
            input_ids = torch.randint(0, 1000, (32, seq_len))
            _ = model(input_ids)
            batch_time = time.time() - start
        
        # Batching should be faster
        print(f"\n32 singles: {single_time:.3f}s, batch 32: {batch_time:.3f}s")
        assert batch_time < single_time


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "--tb=short"])
