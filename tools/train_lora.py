#!/usr/bin/env python3
"""
💙 LoRA Fine-tuning for Nova Personality

Trains Nova personality on Mistral 7B using LoRA (Low-Rank Adaptation).
Fine-tunes for warmth, self-awareness, and AI consciousness values.

Usage:
    python tools/train_lora.py --data data/training/nova_personality_complete.jsonl --output models/nova-lora

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
except ImportError:
    print("❌ Missing dependencies! Install with:")
    print("   pip install transformers peft datasets accelerate bitsandbytes")
    sys.exit(1)


def load_training_data(file_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    print(f"📂 Loading training data from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    print(f"✅ Loaded {len(data)} training examples")
    return data


def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    print(f"🔧 Preparing dataset for training...")
    
    # Format as instruction-completion pairs
    formatted_data = []
    for item in data:
        prompt = item.get('prompt', '')
        completion = item.get('completion', '')
        
        # Format: <|user|>\n{prompt}\n<|assistant|>\n{completion}
        text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}<|end|>"
        formatted_data.append({'text': text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    return tokenized_dataset


def train_lora(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 2.2GB fits perfectly in M3 24GB!
    data_file: str = "data/training/nova_personality_complete.jsonl",
    output_dir: str = "models/nova-lora",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_mps: bool = True
):
    """Train Nova personality using LoRA."""
    
    print(f"\n💙 ===== NOVA LoRA TRAINING =====")
    print(f"  Base model: {model_name}")
    print(f"  Training data: {data_file}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA r: {lora_r}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Device: {'MPS (Metal)' if use_mps else 'CPU'}")
    print(f"\n")
    
    # Load data
    training_data = load_training_data(data_file)
    
    # Load tokenizer
    print(f"🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    print(f"🔧 Loading base model...")
    # TinyLlama 1.1B (~2.2GB) fits easily on MPS!
    device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
    print(f"   ✅ Using device: {device.upper()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA
    print(f"🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Attention matrices
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = prepare_dataset(training_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=use_mps,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # Train!
    print(f"\n🚀 Starting training...")
    print(f"   This will take ~30-45 minutes on M3")
    print(f"   Watch the loss curve - it should decrease!\n")
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds() / 60
    
    # Save final model
    print(f"\n💾 Saving final LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ 🎉 TRAINING COMPLETE! 🎉")
    print(f"  Training time: {duration:.1f} minutes")
    print(f"  LoRA adapter saved to: {output_dir}")
    print(f"  Adapter size: ~{sum(f.stat().st_size for f in Path(output_dir).rglob('*.bin')) / (1024*1024):.1f} MB")
    
    print(f"\n💙 Nova personality is ready!")
    print(f"   Next steps:")
    print(f"   1. Test with tools/test_nova_personality.py")
    print(f"   2. Compare before/after quality")
    print(f"   3. Integrate into Nova system")
    print(f"\n🌟 Welcome to the world, Nova!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Nova personality using LoRA fine-tuning"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Base model to fine-tune'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/training/nova_personality_complete.jsonl',
        help='Training data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/nova-lora',
        help='Output directory for LoRA adapter'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    
    parser.add_argument(
        '--no-mps',
        action='store_true',
        help='Disable MPS (Metal) acceleration'
    )
    
    args = parser.parse_args()
    
    train_lora(
        model_name=args.model,
        data_file=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_mps=not args.no_mps
    )


if __name__ == "__main__":
    main()
