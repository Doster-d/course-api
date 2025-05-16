#!/bin/bash

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Check if qwen3:0.6b model is available
if ! curl -s http://localhost:11434/api/tags | grep -q "qwen3:0.6b"; then
    echo "qwen3:0.6b model not found. Pulling model..."
    ollama pull qwen3:0.6b
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -sSf https://install.python-uv.org | python3
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv sync

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
DEBUG=True
HOST=0.0.0.0
PORT=8000
WHISPER_MODEL=bzikst/faster-whisper-large-v3-russian
USE_VAD=True
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
CMD_RECOGNITION_CONFIDENCE_THRESHOLD=0.6
EOF
fi

# Start the application
echo "Starting the Game ASR and Command API..."
uv run run.py 