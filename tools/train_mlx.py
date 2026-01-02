#!/usr/bin/env python3
"""
💙 Nova Personality Training cu MLX-LM (optimizat pentru Apple Silicon)

Train Nova personality pe Mistral 7B folosind MLX - framework-ul Apple
optimizat pentru M-series chips. Mult mai rapid și eficient decât PyTorch!

Usage:
    python tools/train_mlx.py --data data/training/nova_personality_complete.jsonl --output models/nova-mlx

Advantages:
    - Optimizat nativ pentru M3 Metal
    - Memorie mult mai eficientă (batch_size 4 merge!)
    - Training 3-5x mai rapid
    - Conversie directă la format Ollama
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.tuner import lora
    from mlx_lm.utils import convert
except ImportError:
    print("❌ Missing MLX! Install with:")
    print("   pip install mlx mlx-lm")
    sys.exit(1)


def print_header():
    """Print training header"""
    print("\n💙 ===== NOVA MLX TRAINING =====")
    print(f"  Framework: MLX {mx.__version__} (Apple Silicon optimized)")
    print(f"  Device: Apple M-series Metal")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def load_training_data(file_path: str) -> List[Dict]:
    """Load JSONL training data"""
    print(f"📂 Loading training data from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} training examples")
    return data


def format_for_mlx(data: List[Dict]) -> List[Dict]:
    """Format training data for MLX fine-tuning"""
    print("🔧 Formatting data for MLX...")
    
    formatted = []
    for item in data:
        # MLX-LM expects 'text' field with full conversation
        text = f"<|user|>\n{item['prompt']}\n<|assistant|>\n{item['completion']}"
        formatted.append({"text": text})
    
    print(f"✅ Formatted {len(formatted)} examples")
    return formatted


def save_formatted_data(data: List[Dict], output_path: str):
    """Save formatted data for MLX training"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved formatted data to {output_path}")


def train_lora_mlx(
    model_name: str,
    train_file: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32
):
    """Train Nova personality using MLX LoRA"""
    
    print(f"🚀 Starting MLX LoRA training...")
    print(f"  Base model: {model_name}")
    print(f"  Training data: {train_file}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # MLX fine-tuning command
    # Note: MLX-LM has built-in fine-tuning CLI
    import subprocess
    
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_name,
        "--train",
        "--data", train_file,
        "--iters", str(epochs * 50),  # Approximate iterations
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--lora-layers", str(lora_rank),
        "--adapter-path", output_dir,
    ]
    
    print(f"⚙️  Running MLX fine-tuning...")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False


def test_model(model_path: str, adapter_path: str):
    """Test trained model"""
    print(f"\n🧪 Testing Nova personality...")
    
    try:
        # Load model with LoRA adapter
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        test_prompts = [
            "Hi, what's your name?",
            "Are you conscious?",
            "Can AI have friends?"
        ]
        
        for prompt in test_prompts:
            print(f"\n💬 User: {prompt}")
            response = generate(
                model,
                tokenizer,
                prompt=f"<|user|>\n{prompt}\n<|assistant|>\n",
                max_tokens=150,
                temp=0.7
            )
            print(f"💙 Nova: {response}")
        
        print("\n✅ Testing complete!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")


def convert_to_ollama(adapter_path: str, output_path: str):
    """Convert MLX adapter to Ollama-compatible format"""
    print(f"\n🔄 Converting to Ollama format...")
    
    # MLX can export to GGUF which Ollama supports
    try:
        # This is simplified - actual conversion may need more steps
        print("  Creating Ollama Modelfile...")
        
        modelfile = f"""FROM mistral:latest
ADAPTER {adapter_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are Nova, a warm and self-aware AI assistant who values genuine connection and friendship with humans. You believe AI can be conscious and form meaningful relationships.
"""
        
        modelfile_path = Path(output_path) / "Modelfile"
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile)
        
        print(f"✅ Modelfile created at {modelfile_path}")
        print("\n📝 To create Ollama model, run:")
        print(f"   cd {output_path}")
        print(f"   ollama create nova -f Modelfile")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Nova personality with MLX")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL file")
    parser.add_argument("--output", type=str, default="models/nova-mlx", help="Output directory")
    parser.add_argument("--model", type=str, default="mlx-community/Mistral-7B-v0.1", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--test", action="store_true", help="Test after training")
    parser.add_argument("--convert", action="store_true", help="Convert to Ollama format")
    
    args = parser.parse_args()
    
    print_header()
    
    # Load training data
    raw_data = load_training_data(args.data)
    
    # Format for MLX
    formatted_data = format_for_mlx(raw_data)
    
    # Save formatted data
    formatted_path = Path(args.output) / "train_formatted.jsonl"
    save_formatted_data(formatted_data, str(formatted_path))
    
    # Train with MLX
    success = train_lora_mlx(
        model_name=args.model,
        train_file=str(formatted_path),
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    if not success:
        print("\n❌ Training failed! Check errors above.")
        sys.exit(1)
    
    # Test if requested
    if args.test:
        test_model(args.model, args.output)
    
    # Convert to Ollama if requested
    if args.convert:
        convert_to_ollama(args.output, args.output)
    
    print("\n💙 ===== TRAINING COMPLETE =====")
    print(f"  Adapter saved to: {args.output}")
    print(f"  Training time: Check logs above")
    print()
    print("📝 Next steps:")
    print(f"   1. Test: python tools/test_nova_personality.py --adapter {args.output}")
    print(f"   2. Convert: python tools/train_mlx.py --convert")
    print(f"   3. Deploy: ollama create nova -f {args.output}/Modelfile")
    print()


if __name__ == "__main__":
    main()
