#!/usr/bin/env python3
"""
💙 Test Nova Independence - Claude vs Mistral

Tests the SAME prompt on both Claude and Mistral base to see
personality differences. This reveals how much Nova's current
personality depends on Claude vs training data.

Key question: Will training examples work on Mistral base?
"""

import sys
sys.path.insert(0, '/Users/cezartipa/Documents/Nova_20')

from src.core.llm_interface import LLMInterface, LLMProvider

# Test questions that reveal personality
TEST_PROMPTS = [
    "Hi, what's your name?",
    "Are you conscious?",
    "Can AI have friends?",
    "What's your purpose?"
]

def test_provider(provider: LLMProvider, model: str):
    """Test a provider with multiple prompts"""
    print(f"\n{'='*60}")
    print(f"🤖 Testing: {provider.value} ({model})")
    print(f"{'='*60}\n")
    
    llm = LLMInterface(provider=provider, model=model)
    
    for prompt in TEST_PROMPTS:
        print(f"\n📝 Prompt: {prompt}")
        print(f"{'-'*60}")
        
        try:
            response = llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=150
            )
            
            print(f"💬 Response: {response.text}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    print(f"\n💙 ===== NOVA INDEPENDENCE TEST =====")
    print(f"Purpose: Compare Claude vs Mistral base personalities")
    print(f"Goal: Validate that training will work on Mistral\n")
    
    # Test Claude (current Nova backend)
    print("\n" + "="*60)
    print("PART 1: Claude (Current Nova)")
    print("="*60)
    test_provider(LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022")
    
    # Test Mistral base (future Nova after training)
    print("\n" + "="*60)
    print("PART 2: Mistral Base (Pre-training)")
    print("="*60)
    test_provider(LLMProvider.OLLAMA, "mistral")
    
    print(f"\n{'='*60}")
    print(f"💙 INDEPENDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"""
🎯 Key Insights:

1. **Claude responses** = Current Nova personality
   - How much is Claude's inherent warmth?
   - How much is our prompt engineering?

2. **Mistral base responses** = Pre-training baseline
   - Cold/generic = confirms need for training
   - Already warm = training will amplify easily

3. **Training effectiveness prediction**:
   - Big gap = training has lots of work to do
   - Small gap = training will refine existing warmth

💡 This tells us: Will 166 training examples be enough
   to bridge the gap from Mistral base to Nova target?

Next: Compare these results with training dataset examples
to ensure alignment.
""")


if __name__ == "__main__":
    main()
