# Distributed Training with Ray Train

Ray Train simplifies distributed PyTorch training by handling process group initialization, data distribution, and checkpoint management automatically.

## Files

| File | Description |
|------|-------------|
| `train_ray_ddp.py` | Ray Train with PyTorch DataLoader |
| `train_ray_ddp_with_ray_data.py` | Ray Train with Ray Data for distributed preprocessing |

## Quick Start

```bash
# Using PyTorch DataLoader
python train_ray_ddp.py --num-workers 4 --epochs 3

# Using Ray Data (recommended for large datasets)
python train_ray_ddp_with_ray_data.py --num-workers 4 --epochs 3
```

## What is Ray Train?

[Ray Train](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) is a scalable machine learning library for distributed training. It abstracts away the complexity of:

- **Process group management** - Automatically initializes and manages `torch.distributed` process groups
- **Device placement** - Moves models and data to the correct GPU without manual `.to(device)` calls
- **Distributed sampling** - Handles data partitioning across workers automatically
- **Checkpointing** - Saves checkpoints to persistent shared storage with built-in fault tolerance

### How Ray Train Works Internally

When you call `trainer.fit()`, Ray Train:

1. Spawns the specified number of worker processes across your cluster
2. Initializes a distributed process group (NCCL backend for GPUs)
3. Executes your training function on each worker in parallel
4. Coordinates gradient synchronization through DDP
5. Manages checkpoint storage and metric reporting

## Key Ray Train APIs

Ray Train requires only 3 changes to convert PyTorch training code:

### 1. `prepare_data_loader()`

```python
train_loader = ray.train.torch.prepare_data_loader(train_loader)
```

This function:
- Adds a `DistributedSampler` to partition data across workers
- Moves batches to the correct device automatically
- Handles `sampler.set_epoch()` internally for proper shuffling

If you already have a `DistributedSampler` configured, Ray Train respects your existing setup.

### 2. `prepare_model()`

```python
model = ray.train.torch.prepare_model(model)
```

This function:
- Moves your model to the correct GPU device
- Wraps the model with `DistributedDataParallel`
- Eliminates manual device identification and `.cuda()` calls

### 3. `ray.train.report()`

```python
ray.train.report({"loss": loss, "epoch": epoch}, checkpoint=checkpoint)
```

This function:
- Reports training metrics for monitoring
- Saves checkpoints to persistent shared storage
- Enables fault tolerance - training can resume from the last checkpoint

## TorchTrainer Configuration

```python
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_func,                    # Your training function
    train_loop_config={...},       # Hyperparameters passed to train_func
    scaling_config=ScalingConfig(
        num_workers=4,             # Number of distributed workers
        use_gpu=True,              # Enable GPU training
    ),
    run_config=RunConfig(
        storage_path="/mnt/...",   # Shared storage for checkpoints (required for multi-node)
    ),
)
result = trainer.fit()
```

## What is Ray Data?

[Ray Data](https://docs.ray.io/en/latest/data/data.html) is a scalable data processing library designed for ML workloads. It provides:

- **Streaming execution** - Processes large datasets efficiently without loading everything into memory
- **Heterogeneous compute** - Runs CPU preprocessing separately from GPU training
- **Format flexibility** - Supports Parquet, images, JSON, CSV, audio, video, and more
- **Cloud integration** - Works with S3, GCS, Azure Blob, and other cloud storage

### Why Use Ray Data with Ray Train?

In `train_ray_ddp_with_ray_data.py`, preprocessing runs on separate CPU workers while GPU workers focus on training:

```
┌─────────────────────────────────────────────────────────────┐
│                        Ray Cluster                          │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Ray Data    │    │ Ray Data    │    │ Ray Data    │     │
│  │ Worker (CPU)│    │ Worker (CPU)│    │ Worker (CPU)│     │
│  │ Preprocess  │    │ Preprocess  │    │ Preprocess  │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └────────────┬─────┴─────┬────────────┘             │
│                      ▼           ▼                          │
│              ┌──────────┐ ┌──────────┐                      │
│              │ Train    │ │ Train    │                      │
│              │ Worker   │ │ Worker   │                      │
│              │ (GPU)    │ │ (GPU)    │                      │
│              └──────────┘ └──────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

Benefits:
- **GPU utilization** - GPUs aren't waiting for CPU-bound preprocessing
- **Memory efficiency** - Data streams to workers on-demand, not loaded all at once
- **Scalability** - Add more CPU workers for preprocessing without changing GPU count

### Ray Data API

```python
# Create dataset from various sources
train_ds = ray.data.read_parquet("s3://bucket/data/")
train_ds = ray.data.from_pandas(df)
train_ds = ray.data.read_images("s3://bucket/images/")

# Apply transformations (runs on CPU workers)
train_ds = train_ds.map_batches(preprocess_fn)

# Pass to TorchTrainer
trainer = TorchTrainer(..., datasets={"train": train_ds})

# In training function, get data shard for this worker
train_shard = ray.train.get_dataset_shard("train")
for batch in train_shard.iter_torch_batches(batch_size=32):
    # Training loop
```

## Version Comparison

| Aspect | `train_ray_ddp.py` | `train_ray_ddp_with_ray_data.py` |
|--------|--------------------|---------------------------------|
| Data Loading | PyTorch DataLoader | Ray Data |
| Preprocessing | On GPU workers | On separate CPU workers |
| Memory | Dataset loaded per worker | Streaming, on-demand |
| Best For | Small/medium datasets | Large datasets, CPU-heavy transforms |
| Data Sources | Local files | Any (S3, Parquet, images, etc.) |

## Scaling

Scale to any number of GPUs by changing `--num-workers`:

```bash
# Single node, 4 GPUs
python train_ray_ddp.py --num-workers 4

# Multi-node, 16 GPUs (automatically distributed)
python train_ray_ddp.py --num-workers 16

# Scale preprocessing with Ray Data
python train_ray_ddp_with_ray_data.py --num-workers 16
```

No SSH setup, no `torchrun` commands, no manual coordination required.

## Fault Tolerance

Ray Train provides automatic fault tolerance:

- **Checkpointing** - `ray.train.report()` saves checkpoints to shared storage
- **Worker recovery** - If a worker fails, Ray can restart it and resume from the last checkpoint
- **Graceful degradation** - Training continues even if some workers become unavailable

This is a major advantage over vanilla PyTorch DDP, where any worker failure crashes the entire job.
