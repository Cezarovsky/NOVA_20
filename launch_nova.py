#!/usr/bin/env python3
"""
🚀 NOVA Launch Script

Quick launcher for NOVA Streamlit UI.

Usage:
    ./launch_nova.sh
    
    or
    
    python launch_nova.py
"""

import subprocess
import sys
from pathlib import Path

# Change to Nova_20 directory
nova_dir = Path(__file__).parent
sys.path.insert(0, str(nova_dir))

print("🌟 Launching NOVA UI...")
print(f"📁 Working directory: {nova_dir}")
print("🌐 Starting Streamlit server...\n")

# Run streamlit
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(nova_dir / "src" / "ui" / "streamlit_app.py"),
    "--server.headless", "false"
])
