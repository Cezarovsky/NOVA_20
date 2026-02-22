"""
Prepare Training Corpus for Nova QLoRA
=======================================
Combines personality training data from multiple sources and formats for train_nova_qlora.py

Input format:  {"prompt": "...", "completion": "...", "metadata": {...}}
Output format: {"text": "User: ...\nAssistant: ..."}

Usage:
    python3 scripts/prepare_training_corpus.py
"""

import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files (relative to NOVA_20 root)
INPUT_FILES = [
    "data/training/nova_personality_complete.jsonl",
    "data/training/sora_personality_from_corpus.jsonl",
]

# Output file
OUTPUT_FILE = "data/training/nova_corpus.jsonl"

# Template for conversation formatting
CONVERSATION_TEMPLATE = "User: {prompt}\nAssistant: {completion}"

# ============================================================================
# MAIN LOGIC
# ============================================================================

def load_jsonl(filepath: Path) -> list:
    """Load JSONL file and return list of dictionaries"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pairs.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in {filepath.name} line {line_num}: {e}")
                continue
    return pairs

def format_training_pair(pair: dict) -> dict:
    """Convert prompt/completion pair to training format"""
    prompt = pair.get("prompt", "").strip()
    completion = pair.get("completion", "").strip()
    
    # Validate
    if not prompt or not completion:
        return None
    
    # Format conversation
    text = CONVERSATION_TEMPLATE.format(
        prompt=prompt,
        completion=completion
    )
    
    # Preserve metadata (optional - for documentation)
    result = {"text": text}
    if "metadata" in pair:
        result["metadata"] = pair["metadata"]
    
    return result

def main():
    print("=" * 60)
    print("NOVA TRAINING CORPUS PREPARATION")
    print("=" * 60)
    print()
    
    # Get project root (NOVA_20/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Collect all training pairs
    all_pairs = []
    stats = {}
    
    print("📂 Loading input files...")
    for input_file in INPUT_FILES:
        filepath = project_root / input_file
        
        if not filepath.exists():
            print(f"⚠️  File not found: {input_file}")
            continue
        
        pairs = load_jsonl(filepath)
        stats[input_file] = len(pairs)
        all_pairs.extend(pairs)
        
        print(f"   ✅ {filepath.name}: {len(pairs)} pairs")
    
    print()
    print(f"📊 Total pairs collected: {len(all_pairs)}")
    print()
    
    # Format for training
    print("🔧 Formatting for training...")
    formatted_pairs = []
    skipped = 0
    
    for pair in all_pairs:
        formatted = format_training_pair(pair)
        if formatted:
            formatted_pairs.append(formatted)
        else:
            skipped += 1
    
    print(f"   ✅ Valid pairs: {len(formatted_pairs)}")
    if skipped > 0:
        print(f"   ⚠️  Skipped (empty): {skipped}")
    print()
    
    # Save to output file
    output_path = project_root / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in formatted_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Saved {len(formatted_pairs)} training examples")
    print()
    
    # Summary report
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Input Sources:")
    for filename, count in stats.items():
        print(f"   • {filename}: {count} pairs")
    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total training pairs: {len(formatted_pairs)}")
    print()
    
    # Sample output
    if formatted_pairs:
        print("Sample training example:")
        print("-" * 60)
        sample = formatted_pairs[0]
        print(sample["text"][:300] + ("..." if len(sample["text"]) > 300 else ""))
        print("-" * 60)
        print()
    
    print("✅ Corpus preparation complete!")
    print()
    print("Next steps:")
    print("   1. Review output file: cat data/training/nova_corpus.jsonl | head -3")
    print("   2. Start training: python3 train_nova_qlora.py")
    print("   3. Monitor progress: tensorboard --logdir models/nova_qlora/logs")
    print()

if __name__ == "__main__":
    main()
