"""
Test Ollama Integration în Nova

Quick test pentru noul OllamaProvider
"""

import sys
from pathlib import Path

# Add Nova to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_interface import LLMInterface, LLMProvider

def test_ollama():
    print("🚀 Testing Ollama Integration\n")
    
    # Initialize with Ollama
    llm = LLMInterface(
        provider=LLMProvider.OLLAMA,
        model="mistral"
    )
    
    print(f"✅ LLM Interface initialized")
    print(f"   Provider: {llm.provider}")
    print(f"   Model: {llm.model}\n")
    
    # Test generate
    print("📝 Testing generation...")
    response = llm.generate(
        prompt="Explică foarte scurt ce este inteligența artificială în română.",
        max_tokens=100,
        temperature=0.7
    )
    
    print("\n📊 Response:")
    print(f"   Text: {response.text[:200]}...")
    print(f"   Model: {response.model}")
    print(f"   Provider: {response.provider}")
    print(f"   Tokens: {response.usage['total_tokens']}")
    print(f"   Latency: {response.latency_ms:.0f}ms")
    print(f"   Finish reason: {response.finish_reason}")
    
    print("\n✅ Test successful!")
    print(f"   Ollama is integrated and working on M3!")


if __name__ == '__main__':
    test_ollama()
