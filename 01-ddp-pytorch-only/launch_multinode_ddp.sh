#!/bin/bash
# =============================================================================
# Multi-Node PyTorch DDP Launch Instructions
#
# This script demonstrates what is required to run distributed training
# across multiple machines using vanilla PyTorch DDP.
# =============================================================================

# Configuration (modify these for your setup)
NUM_NODES=2
GPUS_PER_NODE=4
MASTER_ADDR="10.0.0.1"  # IP of the first node
MASTER_PORT=29500

# Training parameters
EPOCHS=${1:-3}
BATCH_SIZE=${2:-128}
LR=${3:-0.001}

cat << 'EOF'
================================================================================
Multi-Node PyTorch DDP Training - Manual Setup Requirements
================================================================================

To run distributed training across multiple machines with vanilla PyTorch,
you need to complete the following steps:

PREREQUISITES
-------------
1. Passwordless SSH access between all nodes
2. Training script accessible on all nodes (via shared filesystem or copied)
3. Same Python environment on all nodes
4. Network connectivity between nodes on the master port

EOF

echo "LAUNCH COMMANDS"
echo "---------------"
echo "Run these commands simultaneously on each node:"
echo ""

for ((i=0; i<NUM_NODES; i++)); do
    echo "# Node $i:"
    echo "torchrun \\"
    echo "    --nnodes=$NUM_NODES \\"
    echo "    --nproc_per_node=$GPUS_PER_NODE \\"
    echo "    --node_rank=$i \\"
    echo "    --master_addr=$MASTER_ADDR \\"
    echo "    --master_port=$MASTER_PORT \\"
    echo "    train_ddp.py --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lr=$LR"
    echo ""
done

cat << 'EOF'
ENVIRONMENT VARIABLES SET BY TORCHRUN
-------------------------------------
- RANK: Global rank of the process (0 to world_size-1)
- LOCAL_RANK: Rank within the current node (0 to nproc_per_node-1)
- WORLD_SIZE: Total number of processes across all nodes
- MASTER_ADDR: IP address of the rank 0 process
- MASTER_PORT: Port for process group communication

KEY CHALLENGES WITH MANUAL DDP
------------------------------
1. Must launch torchrun on each node separately (or use SSH orchestration)
2. All nodes must be ready before training can begin
3. If any node fails, the entire job fails with no recovery
4. Need to manage distributed sampler and set_epoch() manually
5. Requires shared storage or manual script distribution
6. Debugging across nodes is difficult

================================================================================
This complexity is exactly what Ray Train simplifies - see 02-ddp-pytorch-ray/
================================================================================
EOF
