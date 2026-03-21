#!/bin/bash
set -e

echo "🚀 Starting Multi-Agent Code Generation Setup (GPU-Enabled)"

# 1. Update and install basic dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip curl git

# 2. Install Ollama for Linux (GPU support is automatic if drivers are present)
if ! command -v ollama &> /dev/null; then
    echo "🦙 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "🦙 Ollama already installed, skipping..."
fi

# 3. Start Ollama and pull the model
echo "📥 Pulling Qwen2.5-Coder 7B model..."
# Ensure ollama is running in background if not already a service
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > ollama.log 2>&1 &
    sleep 5
fi
ollama pull qwen2.5-coder:7b

# 4. Setup Python Virtual Environment
echo "🐍 Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify GPU Acceleration
echo "🔍 Verifying NVIDIA GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected. Ollama will automatically use it."
else
    echo "⚠️ NVIDIA GPU not detected by nvidia-smi. Check your drivers!"
fi

echo "✨ Setup complete!"
echo "To run the experiments, use:"
echo "source .venv/bin/activate"
echo "python3 scripts/run_all_experiments_smol.py  # For a quick test"
echo "python3 scripts/run_all_experiments.py       # For the full sweep"
