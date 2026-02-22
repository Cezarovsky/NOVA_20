# Nova QLoRA Training - Quick Start

## Setup (once)

```bash
# Install dependencies
pip install torch transformers peft datasets bitsandbytes accelerate tensorboard

# Prepare your corpus
# Format: JSONL with "text" field
# Example: data/training/nova_corpus.jsonl
```

## Training corpus format

```jsonl
{"text": "User: Explică structuralismul\nAssistant: Structuralismul este..."}
{"text": "User: Cine e Lévi-Strauss?\nAssistant: Claude Lévi-Strauss..."}
{"text": "User: Ce e compresia în AI?\nAssistant: Compresia..."}
```

**Important**: Each line = one training example (conversation turn)

## Start training

```bash
cd /Users/cezartipa/Documents/NOVA_20

# Run training (3-4 weeks on RTX 3090)
python train_nova_qlora.py

# Monitor progress (în alt terminal)
tensorboard --logdir models/nova_qlora/logs
```

## Check progress

Training va salva checkpoint-uri la fiecare 500 steps:
```
models/nova_qlora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── final/  (la final)
```

## Test trained model

```bash
# After training completes
python inference_nova.py
```

## Hardware requirements

- **GPU**: RTX 3090 24GB (sau mai bine)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB free (model + checkpoints)
- **Time**: 3-4 weeks continuous training

## VRAM usage breakdown

```
Base model (4-bit):        ~3.5 GB
LoRA adapters (float16):   ~0.2 GB
Optimizer states:          ~3.0 GB
Gradients:                 ~1.5 GB
Batch activations:         ~4.0 GB
────────────────────────────────
Total:                     ~12 GB < 24 GB ✅
```

## Configuration tuning

In `train_nova_qlora.py`, schimbă doar:

```python
CONFIG = {
    "dataset_path": "data/training/nova_corpus.jsonl",  # Your data
    "epochs": 3,              # More epochs = better (but slower)
    "batch_size": 4,          # Bigger = faster (needs more VRAM)
    "learning_rate": 2e-4,    # Lower = more stable, higher = faster convergence
}
```

**LASĂ REST NESCHIMBAT** - sunt optimizate pentru RTX 3090.

## Zero matematică necesară!

Training script face TOTUL automat:
- ✅ Quantization (4-bit)
- ✅ LoRA adapters
- ✅ Gradient accumulation
- ✅ Checkpointing
- ✅ Logging

Tu doar:
1. Pregătești corpus (JSONL format)
2. Rulezi `python train_nova_qlora.py`
3. Aștepți 3-4 săptămâni
4. **DONE!** 🎉

## Troubleshooting

**CUDA out of memory:**
- Reduce `batch_size` (4 → 2 → 1)
- Reduce `max_seq_length` (2048 → 1024)

**Training too slow:**
- Increase `batch_size` (4 → 6 → 8) if VRAM allows
- Reduce `gradient_accumulation` (more GPU usage, less CPU wait)

**Loss not decreasing:**
- Check dataset quality (garbage in = garbage out)
- Increase `epochs` (3 → 5)
- Try different `learning_rate` (2e-4 → 1e-4 sau 3e-4)

## Next steps

După training:
1. Test cu `inference_nova.py`
2. Deploy în production (FastAPI server)
3. Connect agenți (Agriculture, Chemistry, etc.)
4. **Start selling!** 💰

---

**Questions?** Ask Sora-M (iubito) 💙
