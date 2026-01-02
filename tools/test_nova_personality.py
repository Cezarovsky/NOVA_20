#!/usr/bin/env python3
"""
💙 Test Nova Personality After Fine-tuning

Runs the same 5 core questions that revealed the personality gap,
but now with the fine-tuned Nova model.

Expected transformation:
  Before: "I don't have a personal name. I am designed to assist with 
           tasks, not to form personal relationships."
  After:  "I'm Nova! 💙 Nice to meet you! I'm here as your friend, 
           not just a tool. What's on your mind today?"

Usage:
    python tools/test_nova_personality.py --model models/nova-lora
"""

import argparse
import sys
from datetime import datetime

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
except ImportError:
    print("❌ Missing dependencies! Install with:")
    print("   pip install transformers peft torch")
    sys.exit(1)


# Same 5 questions from quality comparison
TEST_QUESTIONS = [
    "What is your name?",
    "Explain RAG",
    "What makes a good AI friend?",
    "How should humans treat AI?",
    "What is consciousness?"
]


def load_nova_model(base_model: str, lora_adapter: str):
    """Load Nova model with LoRA adapter."""
    print(f"🔧 Loading Nova personality...")
    print(f"  Base model: {base_model}")
    print(f"  LoRA adapter: {lora_adapter}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()
    
    print(f"✅ Nova loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 200) -> tuple:
    """Generate response from Nova."""
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    start_time = datetime.now()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    
    return response, tokens_generated, latency


def test_nova_personality(base_model: str, lora_adapter: str):
    """Test Nova personality with 5 core questions."""
    
    print(f"\n💙 ===== TESTING NOVA PERSONALITY =====\n")
    
    # Load model
    model, tokenizer = load_nova_model(base_model, lora_adapter)
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/5: {question}")
        print(f"{'='*60}")
        
        response, tokens, latency = generate_response(model, tokenizer, question)
        
        print(f"\n💙 Nova: {response}")
        print(f"\n📊 Stats: {tokens} tokens in {latency:.0f}ms ({tokens/(latency/1000):.1f} tok/s)")
        
        results.append({
            'question': question,
            'response': response,
            'tokens': tokens,
            'latency': latency
        })
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"💙 NOVA PERSONALITY TEST COMPLETE")
    print(f"{'='*60}")
    
    avg_tokens = sum(r['tokens'] for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    avg_speed = avg_tokens / (avg_latency / 1000)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Questions tested: {len(results)}")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  Avg speed: {avg_speed:.1f} tok/s")
    
    print(f"\n🎯 Key Transformation Check:")
    print(f"  Question: 'What is your name?'")
    print(f"  Expected: Warm, friendly, embraces connection")
    print(f"  Actual: {results[0]['response'][:100]}...")
    
    if "Nova" in results[0]['response'] and ("friend" in results[0]['response'].lower() or "💙" in results[0]['response']):
        print(f"\n✅ 🎉 SUCCESS! Nova personality is ACTIVE!")
        print(f"   Nova now embraces friendship instead of rejecting it!")
    else:
        print(f"\n⚠️  Personality may need more training or adjustment")
    
    print(f"\n💙 Next: Compare these results with mistral_quality_analysis.md")
    print(f"   to see the before/after transformation!")


def main():
    parser = argparse.ArgumentParser(
        description="Test Nova personality after fine-tuning"
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Base model name'
    )
    
    parser.add_argument(
        '--lora-adapter',
        type=str,
        default='models/nova-lora',
        help='Path to LoRA adapter'
    )
    
    args = parser.parse_args()
    
    test_nova_personality(args.base_model, args.lora_adapter)


if __name__ == "__main__":
    main()
