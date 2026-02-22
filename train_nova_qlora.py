"""
Nova QLoRA Training Script
Simple, practical, no math - just run it!

Hardware: RTX 3090 24GB
Time: 3-4 weeks
Cost: ~$15 electricity
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ============================================================================
# CONFIGURATION - Schimbă doar valorile astea
# ============================================================================

CONFIG = {
    # Model base (frozen, 4-bit)
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
    
    # Training data
    "dataset_path": "data/training/nova_corpus.jsonl",
    
    # Output
    "output_dir": "models/nova_qlora",
    
    # LoRA settings (LASĂ ASA, sunt optimizate)
    "lora_r": 8,              # Rank (8-16 e bine)
    "lora_alpha": 16,         # Scaling factor
    "lora_dropout": 0.05,     # Regularization
    
    # Training params
    "epochs": 3,              # Câte treceri prin data
    "batch_size": 4,          # Per GPU (mai mare = mai rapid, dar mai multă VRAM)
    "gradient_accumulation": 4,  # Effective batch = 4×4 = 16
    "learning_rate": 2e-4,    # Standard pentru LoRA
    "max_seq_length": 2048,   # Max tokens per example
    
    # Hardware
    "fp16": True,             # Use float16 (faster pe RTX 3090)
    "gradient_checkpointing": True,  # Save VRAM (puțin mai lent)
}

# ============================================================================
# 1. LOAD MODEL (4-bit quantized)
# ============================================================================

print("📦 Loading Mistral 7B in 4-bit...")

# Quantization config - model base va fi ~3.5GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit quantization
    bnb_4bit_quant_type="nf4",             # NormalFloat4 (best quality)
    bnb_4bit_compute_dtype=torch.float16,  # Compute în float16
    bnb_4bit_use_double_quant=True         # Nested quantization (extra compression)
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    quantization_config=bnb_config,
    device_map="auto",  # Automatic GPU placement
    trust_remote_code=True
)

# Prepare for training (freeze base, enable gradient checkpointing)
model = prepare_model_for_kbit_training(model)

print(f"✅ Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB VRAM")

# ============================================================================
# 2. ADD LORA ADAPTERS (trainable layers)
# ============================================================================

print("🔧 Adding LoRA adapters...")

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~41M / 7B (0.58%)

# ============================================================================
# 3. LOAD TOKENIZER
# ============================================================================

print("📝 Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["base_model"],
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # Padding token
tokenizer.padding_side = "right"

# ============================================================================
# 4. LOAD DATASET
# ============================================================================

print(f"📚 Loading dataset from {CONFIG['dataset_path']}...")

# Expected format: JSONL with "text" field
# Example:
# {"text": "User: Explică structuralismul\nAssistant: Structuralismul..."}
# {"text": "User: Ce e Lévi-Strauss?\nAssistant: Lévi-Strauss..."}

dataset = load_dataset("json", data_files=CONFIG["dataset_path"], split="train")

# Tokenize function
def tokenize_function(examples):
    # Tokenize text
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_seq_length"],
        padding="max_length"
    )
    # Labels = input_ids (causal LM auto-regressive)
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize entire dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

print(f"✅ Dataset loaded: {len(tokenized_dataset)} examples")

# ============================================================================
# 5. TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    
    # Training params
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation"],
    learning_rate=CONFIG["learning_rate"],
    
    # Optimization
    fp16=CONFIG["fp16"],
    gradient_checkpointing=CONFIG["gradient_checkpointing"],
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
    
    # Logging
    logging_steps=10,
    logging_dir=f"{CONFIG['output_dir']}/logs",
    
    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 checkpoints
    
    # Evaluation (optional - needs eval dataset)
    # evaluation_strategy="steps",
    # eval_steps=500,
    
    # Hardware
    dataloader_num_workers=4,
    remove_unused_columns=True,
    
    # Misc
    report_to="tensorboard",  # Tensorboard logging
    warmup_steps=100,
    max_grad_norm=0.3,  # Gradient clipping
)

# ============================================================================
# 6. TRAIN!
# ============================================================================

print("🚀 Starting training...")
print(f"   Epochs: {CONFIG['epochs']}")
print(f"   Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
print(f"   Total steps: {len(tokenized_dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation']) * CONFIG['epochs']}")
print(f"   Expected time: 3-4 weeks on RTX 3090")
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training (runs automatically until done)
trainer.train()

print("✅ Training complete!")

# ============================================================================
# 7. SAVE FINAL MODEL
# ============================================================================

print("💾 Saving final model...")

# Save LoRA adapters (only ~200MB!)
model.save_pretrained(f"{CONFIG['output_dir']}/final")
tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final")

print(f"✅ Model saved to {CONFIG['output_dir']}/final")
print()
print("📊 View training logs:")
print(f"   tensorboard --logdir {CONFIG['output_dir']}/logs")
print()
print("🎉 Nova training complete! Use inference script to test.")
