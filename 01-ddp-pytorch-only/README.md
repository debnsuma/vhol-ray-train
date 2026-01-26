# Vanilla PyTorch DDP Training

This folder demonstrates distributed training using **PyTorch's DistributedDataParallel (DDP)** without any additional frameworks.

## What is DistributedDataParallel?

From the [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html):

> DistributedDataParallel (DDP) implements data parallelism at the module level. DDP uses collective communications in the `torch.distributed` package to synchronize gradients and buffers. Each process maintains its own copy of the model, and DDP handles gradient synchronization automatically during the backward pass.

### How DDP Works

1. **Each process has its own model copy** - Unlike DataParallel, DDP creates separate processes (not threads), each with its own Python interpreter
2. **Data is partitioned across processes** - The `DistributedSampler` ensures each GPU processes a unique subset of the data
3. **Gradients are synchronized automatically** - During `loss.backward()`, DDP uses NCCL to average gradients across all processes
4. **All models stay in sync** - After each optimizer step, all model copies have identical weights

### Key Concepts

| Term | Description |
|------|-------------|
| **rank** | Global process identifier (0 to world_size-1) |
| **local_rank** | Process identifier within a single node (0 to nproc_per_node-1) |
| **world_size** | Total number of processes across all nodes |
| **NCCL** | NVIDIA Collective Communications Library - optimized for GPU communication |

## Files

| File | Description |
|------|-------------|
| `train_ddp.py` | DDP training script for ResNet18 on MNIST |
| `launch_multinode_ddp.sh` | Prints instructions for multi-node launch |

## Single-Node Training

For training on a single machine with multiple GPUs:

```bash
# Train on 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py --epochs 3 --batch-size 128 --lr 0.001
```

## Multi-Node Training

For training across multiple machines, run `./launch_multinode_ddp.sh` to see the required setup and commands:

```bash
./launch_multinode_ddp.sh
```

This will print the `torchrun` commands needed for each node. The key parameters are:

- `--nnodes`: Total number of machines
- `--nproc_per_node`: GPUs per machine
- `--node_rank`: Index of the current machine (0, 1, 2, ...)
- `--master_addr`: IP address of node 0
- `--master_port`: Port for inter-process communication

## Manual Steps Required

When using vanilla PyTorch DDP, you must handle:

1. **Process group initialization** - Call `dist.init_process_group()` at start, `dist.destroy_process_group()` at end
2. **Distributed sampler** - Create `DistributedSampler` and pass it to the DataLoader
3. **Epoch shuffling** - Call `sampler.set_epoch(epoch)` each epoch for proper randomization
4. **Model wrapping** - Wrap model with `DistributedDataParallel(model, device_ids=[local_rank])`
5. **Multi-node coordination** - Launch `torchrun` on each node with correct parameters
6. **Infrastructure** - Set up passwordless SSH, shared storage, and network connectivity

