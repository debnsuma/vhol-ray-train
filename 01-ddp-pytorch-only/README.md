# Vanilla PyTorch DDP Training

Shows the complexity of distributed training WITHOUT Ray Train.

## Files

| File | Description |
|------|-------------|
| `train_ddp.py` | DDP training script (~100 lines) |
| `launch_multinode_ddp.sh` | Multi-node launcher |
| `run_multinode_ddp.py` | Orchestration script |

## Run Multi-Node Training

```bash
# Run on 2 nodes (8 GPUs total)
./launch_multinode_ddp.sh 3 128 0.001
# Args: epochs, batch_size, learning_rate
```

## What the Launcher Does

1. Discovers GPU nodes dynamically
2. Copies training script to shared storage
3. Coordinates MASTER_ADDR across nodes
4. Launches torchrun on each node simultaneously
5. Waits for completion

## Pain Points Demonstrated

1. **Requires orchestration** - Can't just run a single command
2. **Manual setup/cleanup** - `init_process_group()` / `destroy_process_group()`
3. **DistributedSampler** - Must create manually
4. **sampler.set_epoch()** - Must call every epoch or shuffling breaks
5. **DDP wrapping** - Manual `DistributedDataParallel(model)`
6. **Shared storage** - Script must be accessible from all nodes
7. **No fault tolerance** - If one node fails, everything fails

## Compare to Ray Train

```bash
# Ray Train - ONE command, same result!
python train_ray_ddp.py --num-workers 8 --epochs 3
```

No launcher script needed. No orchestration code. Just works.
