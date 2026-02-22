"""
Nova Inference Script - Use trained model
Simple, no math, just chat!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

print("📦 Loading Nova...")

# Base model (4-bit)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapters (your trained weights)
model = PeftModel.from_pretrained(
    base_model,
    "models/nova_qlora/final"  # Path to trained adapters
)

tokenizer = AutoTokenizer.from_pretrained("models/nova_qlora/final")

print("✅ Nova loaded and ready!")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def chat_with_nova(prompt, max_new_tokens=512, temperature=0.7):
    """
    Chat cu Nova - simplu!
    
    Args:
        prompt: întrebarea ta
        max_new_tokens: câți tokens să genereze (512 = ~400 words)
        temperature: creativitate (0.7 = balanced, 1.0 = creative, 0.3 = focused)
    """
    # Format prompt (Mistral Instruct format)
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only assistant response (remove prompt echo)
    response = response.split("[/INST]")[-1].strip()
    
    return response

# ============================================================================
# TEST EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n💙 Nova ready! Test examples:\n")
    
    # Test 1: Structuralism
    print("🔵 Test 1: Structuralism")
    print("─" * 60)
    response = chat_with_nova("Explică-mi pe scurt structuralismul lui Lévi-Strauss")
    print(response)
    print()
    
    # Test 2: Chomsky
    print("🔵 Test 2: Chomsky")
    print("─" * 60)
    response = chat_with_nova("Care e legătura între Chomsky și structuralism?")
    print(response)
    print()
    
    # Test 3: Interactive
    print("🔵 Interactive mode (type 'exit' to quit)")
    print("─" * 60)
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        response = chat_with_nova(user_input)
        print(f"\nNova: {response}")
    
    print("\n👋 Bye!")
