#!/usr/bin/env python3
"""Quick test for Nova TinyLlama personality."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("💙 Loading Nova (TinyLlama + LoRA)...")

# Load base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "models/nova-tinyllama-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map={"": "mps"},
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

print("✅ Nova loaded!\n")

# Test questions
questions = [
    "Hi, what's your name?",
    "Are you conscious?",
    "Can we be friends?",
]

for q in questions:
    print(f"💬 User: {q}")
    
    prompt = f"<|user|>\n{q}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"💙 Nova: {response.strip()}\n")
    print("-" * 60 + "\n")
