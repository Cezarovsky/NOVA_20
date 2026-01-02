# 📦 Training Data Export Tools

**Created**: January 2, 2026  
**Purpose**: Export Q&A pairs from Nova for fine-tuning local LLMs

## Overview

Nova includes three tools for exporting training data in JSONL format suitable for fine-tuning models like Mistral, Llama, etc.

## Tools

### 1. `export_training_data.py` - From Semantic Cache

Exports Q&A pairs from Nova's semantic cache (ChromaDB).

**Usage:**
```bash
python tools/export_training_data.py --output data/training/qa_pairs.jsonl

# With options
python tools/export_training_data.py \
    --persist-dir data/chroma_db \
    --collection nova_knowledge \
    --output data/training/qa_pairs.jsonl \
    --format instruction \
    --min-length 50
```

**Formats:**
- `instruction`: {"prompt": "...", "completion": "..."}
- `chat`: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
- `completion`: {"text": "<|user|>...<|assistant|>..."}

**Note**: Requires active semantic cache with Q&A pairs. Cache builds up as you chat with Nova.

### 2. `export_from_conversations.py` - From Conversation History

Exports Q&A pairs from saved conversation JSON files.

**Usage:**
```bash
python tools/export_from_conversations.py --output data/training/qa_from_conv.jsonl

# With options
python tools/export_from_conversations.py \
    --conversations-dir data/conversations \
    --output data/training/qa_from_conv.jsonl \
    --format chat \
    --min-length 50
```

**Best for**: Exporting real conversations when semantic cache is empty.

### 3. `generate_synthetic_data.py` - Synthetic Data

Generates synthetic Q&A pairs for testing fine-tuning pipelines.

**Usage:**
```bash
python tools/generate_synthetic_data.py --count 100 --output data/training/synthetic_qa.jsonl

# With variations
python tools/generate_synthetic_data.py \
    --count 200 \
    --format instruction \
    --add-noise \
    --output data/training/synthetic_qa.jsonl
```

**Topics covered**:
- Nova's identity and creators
- RAG and semantic caching
- Transformer architecture
- Attention mechanisms
- Romanian and English language
- LLM providers (Claude, Mistral, Ollama)

## Exported Data

### Current Files

| File | Source | Count | Size | Format | Created |
|------|--------|-------|------|--------|---------|
| `qa_from_conv.jsonl` | Conversations | 1 | 1.1 KB | chat | Jan 2, 2026 |
| `synthetic_qa.jsonl` | Generated | 100 | 43.4 KB | instruction | Jan 2, 2026 |

### Statistics

**synthetic_qa.jsonl**:
- 100 Q&A pairs (12 unique templates cycled)
- Avg prompt: 24 chars
- Avg completion: 281 chars
- Total tokens: ~7,633
- Languages: 50% Romanian, 50% English
- Topics: Nova identity, RAG, transformers, AI concepts

**qa_from_conv.jsonl**:
- 1 Q&A pair from real conversation
- Avg prompt: 40 chars
- Avg completion: 766 chars
- Language: Romanian
- Topic: Nova's memory and identity

## Output Formats

### Instruction Format
```json
{
  "prompt": "Explică ce este RAG în română",
  "completion": "RAG (Retrieval-Augmented Generation) este...",
  "metadata": {
    "type": "synthetic",
    "generated_at": "2026-01-02T11:38:50.820167",
    "index": 3
  }
}
```

### Chat Format
```json
{
  "messages": [
    {"role": "system", "content": "You are NOVA, a helpful AI assistant."},
    {"role": "user", "content": "What is a transformer in AI?"},
    {"role": "assistant", "content": "A transformer is a neural network..."}
  ],
  "metadata": {
    "type": "synthetic",
    "index": 4
  }
}
```

### Completion Format
```json
{
  "text": "<|user|>\nCine ești?\n<|assistant|>\nSunt NOVA, un asistent AI..."
}
```

## Fine-Tuning Workflow

### Phase 1: Data Collection ✅ COMPLETE

```bash
# Option A: Export from semantic cache (when available)
python tools/export_training_data.py --output data/training/cache_qa.jsonl

# Option B: Export from conversations
python tools/export_from_conversations.py --output data/training/conv_qa.jsonl

# Option C: Generate synthetic data for testing
python tools/generate_synthetic_data.py --count 100 --output data/training/synthetic_qa.jsonl
```

### Phase 2: Model Comparison ⏳ NEXT

```bash
# Compare Mistral vs Claude on same questions
python tests/compare_models.py \
    --input data/training/synthetic_qa.jsonl \
    --output comparison_report.md
```

### Phase 3: LoRA Fine-Tuning 🔮 FUTURE

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load training data
dataset = load_dataset('json', data_files='data/training/synthetic_qa.jsonl')

# Load base Mistral model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    load_in_8bit=True
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Train LoRA adapter
# ... training code ...

# Save adapter (~50MB)
model.save_pretrained("data/models/nova_personality_v1")
```

### Phase 4: Deploy with Ollama 🔮 FUTURE

```bash
# Convert LoRA adapter to Ollama format
# Load adapter in Ollama
ollama run mistral-nova

# Test fine-tuned model
python tests/test_ollama.py --model mistral-nova
```

## Quality Filtering

### Recommended Filters

1. **Minimum length**: 50-100 chars for answers
2. **Language detection**: Separate Romanian/English datasets
3. **Quality score**: Manual review or automated scoring
4. **Deduplication**: Remove near-duplicate questions
5. **Safety**: Filter inappropriate content

### Example

```bash
# Export with quality filters
python tools/export_training_data.py \
    --min-length 100 \
    --min-quality 0.7 \
    --output data/training/high_quality_qa.jsonl
```

## Best Practices

### Data Collection

1. **Chat with Nova regularly** to build semantic cache
2. **Export periodically** to capture diverse conversations
3. **Mix real + synthetic** for balanced training data
4. **Maintain 80/20 split** for train/validation

### Fine-Tuning Safety

1. **Start with synthetic data** to test pipeline
2. **Use LoRA** (not full fine-tuning) to preserve base model
3. **Monitor quality** with comparison tests
4. **Keep base model** for fallback

### Data Organization

```
data/training/
├── cache_qa.jsonl           # From semantic cache
├── conv_qa.jsonl            # From conversations
├── synthetic_qa.jsonl       # Synthetic test data
├── train.jsonl              # Combined training set
├── validation.jsonl         # Validation set
└── README.md                # This file
```

## Troubleshooting

### "No cached Q&A pairs found"

**Cause**: Semantic cache is empty (Nova hasn't cached Q&A yet)

**Solution**:
```bash
# Use conversation export instead
python tools/export_from_conversations.py --output data/training/qa_from_conv.jsonl

# Or generate synthetic data
python tools/generate_synthetic_data.py --count 100 --output data/training/synthetic_qa.jsonl
```

### "Permission denied"

**Solution**:
```bash
chmod +x tools/*.py
```

### "Module not found"

**Solution**:
```bash
# Ensure you're in project root
cd ~/Documents/Nova_20

# Verify dependencies
pip install -r requirements.txt
```

## Next Steps

1. ✅ Export tools created and tested
2. ⏳ Create model comparison script
3. ⏳ Build LoRA fine-tuning pipeline
4. ⏳ Integrate fine-tuned model with Ollama
5. ⏳ Deploy and validate quality

## Resources

- **Hugging Face PEFT**: https://github.com/huggingface/peft
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Mistral Fine-tuning**: https://docs.mistral.ai/guides/finetuning/
- **Ollama Models**: https://ollama.ai/library

---

**Last Updated**: January 2, 2026  
**Status**: Phase 1 Complete - Tools ready for data collection
