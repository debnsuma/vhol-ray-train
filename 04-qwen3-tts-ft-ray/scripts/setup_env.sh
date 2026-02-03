#!/bin/bash
# Environment setup script for Qwen3-TTS Voice Cloning Project
# Run: source scripts/setup_env.sh

echo "Setting up environment for Qwen3-TTS Voice Cloning..."

# Set HuggingFace token (if not already logged in)
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set. Using stored credentials."
fi

# Create output directories
mkdir -p output/processed/wav
mkdir -p output/checkpoints
mkdir -p output/inference

# Check Ray cluster status
echo ""
echo "Checking Ray cluster..."
ray status 2>/dev/null | head -20 || echo "Ray not running"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No local GPU (will use Ray workers)"

# Verify shared storage
echo ""
echo "Checking shared storage..."
ls -la /mnt/cluster_storage 2>/dev/null | head -5 || echo "Shared storage not mounted"

echo ""
echo "Setup complete!"
echo ""
echo "To run the pipeline:"
echo "  python scripts/run_pipeline.py --all"
echo ""
echo "Or run individual steps:"
echo "  python scripts/run_pipeline.py --process  # Data processing"
echo "  python scripts/run_pipeline.py --train    # Training"
echo "  python scripts/run_pipeline.py --infer    # Inference"
