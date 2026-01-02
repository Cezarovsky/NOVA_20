#!/bin/bash
# Desktop Ubuntu Setup Script - Sora Local Clone
# Run on fresh Ubuntu 22.04 installation

set -e

echo "💙 ===== SORA LOCAL CLONE SETUP ====="
echo "Setting up Ubuntu desktop as Sora server..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essentials
echo "🔧 Installing essential tools..."
sudo apt install -y \
    openssh-server \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    htop \
    tmux \
    curl \
    wget \
    build-essential

# Configure SSH
echo "🔐 Configuring SSH server..."
sudo systemctl enable ssh
sudo systemctl start ssh

# Get IP
DESKTOP_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -1)
echo "✅ SSH server running on: $DESKTOP_IP"

# Create workspace
echo "📁 Creating Nova workspace..."
mkdir -p ~/Documents/Nova_20
cd ~/Documents/Nova_20

# Setup Python venv
echo "🐍 Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch CPU (Nvidia 940 too old for modern CUDA)
echo "🔥 Installing PyTorch (CPU version for i9 64GB)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML dependencies
echo "📚 Installing ML libraries..."
pip install transformers accelerate peft bitsandbytes datasets

# Install API server dependencies
echo "🌐 Installing FastAPI for Sora API server..."
pip install fastapi uvicorn pydantic

# Clone repository (assuming you'll push to GitHub)
# echo "📥 Cloning Nova repository..."
# git clone <YOUR_REPO_URL> .

# Create systemd service for Sora API
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/sora-api.service > /dev/null <<EOF
[Unit]
Description=Sora Local API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/Documents/Nova_20
Environment="PATH=$HOME/Documents/Nova_20/venv/bin"
ExecStart=$HOME/Documents/Nova_20/venv/bin/python tools/sora_api_server.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Sora API service created (not started yet)"

# Create SSH config helper for Mac
echo "📝 Creating SSH config for Mac..."
cat > ~/sora_ssh_config.txt <<EOF
# Add this to your Mac ~/.ssh/config:

Host sora-desktop
    HostName $DESKTOP_IP
    User $USER
    Port 22
    LocalForward 8000 localhost:8000  # Sora API
    LocalForward 8501 localhost:8501  # Streamlit UI
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Usage on Mac:
#   ssh sora-desktop
#   # Then Nova can use http://localhost:8000/v1/messages
EOF

echo ""
echo "🎉 ===== SETUP COMPLETE ====="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. On Mac, copy SSH key:"
echo "   ssh-copy-id $USER@$DESKTOP_IP"
echo ""
echo "2. On Mac, add SSH config:"
echo "   cat ~/Documents/Nova_20/sora_ssh_config.txt >> ~/.ssh/config"
echo ""
echo "3. Train Sora model:"
echo "   cd ~/Documents/Nova_20"
echo "   source venv/bin/activate"
echo "   python tools/train_lora.py \\"
echo "     --model mistralai/Mistral-7B-v0.1 \\"
echo "     --data data/training/sora_personality.jsonl \\"
echo "     --output models/sora-lora \\"
echo "     --epochs 3 --batch-size 2"
echo ""
echo "4. Start Sora API server:"
echo "   sudo systemctl start sora-api"
echo "   sudo systemctl enable sora-api  # Auto-start on boot"
echo ""
echo "5. On Mac, configure Nova:"
echo "   export ANTHROPIC_BASE_URL=http://localhost:8000"
echo "   ssh sora-desktop  # This forwards port 8000"
echo ""
echo "✨ Desktop IP: $DESKTOP_IP"
echo "✨ SSH config saved to: ~/sora_ssh_config.txt"
