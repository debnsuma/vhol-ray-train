#!/bin/bash
# =============================================================================
# Multi-Node DDP Launcher
#
# Uses Ray ONLY for orchestration - the actual training is vanilla PyTorch DDP!
# This demonstrates that multi-node DDP requires orchestration infrastructure.
# =============================================================================

set -e

EPOCHS=${1:-3}
BATCH_SIZE=${2:-128}
LR=${3:-0.001}

echo "============================================================"
echo "Launching Multi-Node Vanilla PyTorch DDP Training"
echo "============================================================"
echo ""
echo "Arguments: epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"
echo ""

# Copy training script to shared storage
mkdir -p /mnt/cluster_storage/vhol-ddp
cp "$(dirname "$0")/train_ddp.py" /mnt/cluster_storage/vhol-ddp/

# Run the Python launcher
python3 "$(dirname "$0")/run_multinode_ddp.py" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR"
