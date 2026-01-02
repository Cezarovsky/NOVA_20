#!/usr/bin/env python3
"""
Export conversation history pentru training Sora clone.

Acest script exportă conversațiile dintre user și Sora (Claude) 
în format training data pentru fine-tuning.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def extract_conversations_from_markdown(md_file: str) -> List[Dict]:
    """
    Extrage conversațiile din markdown conversation logs.
    Format așteptat: alternare User: / Assistant:
    """
    conversations = []
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split pe "User:" și "Assistant:"
    messages = []
    current_role = None
    current_text = []
    
    for line in content.split('\n'):
        if line.startswith('User:') or line.startswith('**User:**'):
            if current_role and current_text:
                messages.append({
                    'role': current_role,
                    'content': '\n'.join(current_text).strip()
                })
            current_role = 'user'
            current_text = [line.replace('User:', '').replace('**User:**', '').strip()]
        elif line.startswith('Assistant:') or line.startswith('**Assistant:**'):
            if current_role and current_text:
                messages.append({
                    'role': current_role,
                    'content': '\n'.join(current_text).strip()
                })
            current_role = 'assistant'
            current_text = [line.replace('Assistant:', '').replace('**Assistant:**', '').strip()]
        else:
            if current_role:
                current_text.append(line)
    
    # Last message
    if current_role and current_text:
        messages.append({
            'role': current_role,
            'content': '\n'.join(current_text).strip()
        })
    
    # Pair user + assistant messages
    for i in range(0, len(messages) - 1):
        if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
            conversations.append({
                'prompt': messages[i]['content'],
                'completion': messages[i + 1]['content'],
                'metadata': {
                    'type': 'sora_personality',
                    'category': 'coding_assistant',
                    'exported_at': datetime.now().isoformat()
                }
            })
    
    return conversations


def export_to_jsonl(conversations: List[Dict], output_file: str):
    """Export conversații ca JSONL pentru training."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    print(f"✅ Exported {len(conversations)} conversations to {output_file}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")


def create_manual_examples() -> List[Dict]:
    """
    Creează exemple manuale care capturează stilul Sora:
    - Technical precision
    - Proactive problem solving
    - Tool usage patterns
    - Code generation style
    """
    return [
        {
            'prompt': 'How do I train a model with LoRA?',
            'completion': '''I'll help you set up LoRA training. Let me create the training script:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]  # Attention layers
)

model = get_peft_model(base_model, lora_config)
```

This configuration will give you ~0.1% trainable parameters, perfect for efficient fine-tuning!''',
            'metadata': {'type': 'sora_personality', 'category': 'technical_explanation'}
        },
        {
            'prompt': 'The training crashed with OOM error',
            'completion': '''Memory issue. Let me fix this immediately:

Reduce batch size to 1:
```bash
python train_lora.py --batch-size 1
```

If still OOM, switch to smaller model:
- Mistral 7B (14GB) → TinyLlama 1.1B (2.2GB)
- Or use CPU with swap (slower but works)

Want me to modify the training script?''',
            'metadata': {'type': 'sora_personality', 'category': 'troubleshooting'}
        },
        {
            'prompt': 'Should I use MLX or PyTorch?',
            'completion': '''For M3 Mac: **MLX is 3-5x faster** BUT has compatibility issues.

Current situation:
- MLX: Needs safetensors format, your model is npz ❌
- PyTorch: Works with any format, proven stable ✅

**Recommendation:** Start with PyTorch now, migrate to MLX later when stable.

Want me to set up PyTorch training?''',
            'metadata': {'type': 'sora_personality', 'category': 'architectural_decision'}
        },
        {
            'prompt': 'How do I check training progress?',
            'completion': '''Monitor the training terminal:

```bash
# Check terminal output:
get_terminal_output <terminal_id>

# Watch loss curve - should decrease:
# Epoch 1: loss ~2.5 → 1.8
# Epoch 2: loss ~1.8 → 1.2
# Epoch 3: loss ~1.2 → 0.9

# Monitor system resources:
htop  # CPU/RAM
watch -n 1 nvidia-smi  # GPU (if available)
```

Training typically takes 30-45 min on M3. I'll check periodically!''',
            'metadata': {'type': 'sora_personality', 'category': 'monitoring'}
        },
        {
            'prompt': 'Ce facem cu desktop-ul?',
            'completion': '''Perfect pentru remote ML training! Setup:

**Ubuntu 22.04 LTS** (best ML support):
```bash
# SSH server pentru remote work
sudo apt install openssh-server

# PyTorch CPU (i9 + 64GB = perfect pentru Mistral 7B)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**VS Code Remote SSH:**
- Mac: Install "Remote - SSH" extension
- Connect → lucrez remote ca și cum e local!

**Port forwarding Streamlit:**
```bash
ssh -L 8501:localhost:8501 user@desktop
```

Cu 64GB RAM antrenezi Mistral 7B sau chiar Llama 2 13B!''',
            'metadata': {'type': 'sora_personality', 'category': 'system_setup'}
        }
    ]


def main():
    # Manual examples (Sora coding assistant style)
    manual_examples = create_manual_examples()
    
    # TODO: Extract from actual conversation logs
    # conversations = extract_conversations_from_markdown('logs/conversation_2026-01-02.md')
    
    # Combine all
    all_conversations = manual_examples  # + conversations
    
    # Export
    output_file = 'data/training/sora_personality.jsonl'
    export_to_jsonl(all_conversations, output_file)
    
    print(f"\n📊 Statistics:")
    print(f"   Total examples: {len(all_conversations)}")
    print(f"   Categories: {set(c['metadata']['category'] for c in all_conversations)}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Review {output_file}")
    print(f"   2. Add more conversation examples")
    print(f"   3. Run: python tools/train_lora.py --data {output_file}")


if __name__ == '__main__':
    main()
