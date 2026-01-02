#!/usr/bin/env python3
"""Download MLX model from HuggingFace"""

from huggingface_hub import snapshot_download
import sys

# Use PyTorch model by default (MLX will convert it)
model_id = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.2"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "models/mistral-7b-instruct-pytorch"

print(f"📥 Downloading {model_id}...")
print(f"   Output: {output_dir}")
print("   This may take 2-3 minutes (~3GB)")

try:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    print(f"\n✅ Downloaded successfully to: {path}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    sys.exit(1)
