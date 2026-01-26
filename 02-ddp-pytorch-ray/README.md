# DDP with Ray Train

Distributed training made simple - just 3 changes from vanilla PyTorch!

## Quick Start

```bash
# Version 1: Standard DataLoader
cd version1-without-ray-data
python train_ray_ddp.py --num-workers 4 --epochs 3

# Version 2: With Ray Data
cd version2-with-ray-data
python train_ray_ddp_ray_data.py --num-workers 4 --epochs 3
```

## The 3 Key Changes

```python
# 1. Wrap DataLoader (replaces DistributedSampler)
train_loader = ray.train.torch.prepare_data_loader(train_loader)

# 2. Wrap Model (replaces DDP wrapping)
model = ray.train.torch.prepare_model(model)

# 3. Report Metrics (replaces manual logging)
ray.train.report({"loss": loss})
```

## Version Comparison

| | Version 1 | Version 2 |
|---|-----------|-----------|
| Data Loading | PyTorch DataLoader | Ray Data |
| Preprocessing | On GPU workers | On CPU workers |
| Best For | Simple datasets | Large datasets, CPU-heavy preprocessing |

## Scaling

Just change `num_workers`:
```bash
python train_ray_ddp.py --num-workers 8  # Uses 8 GPUs across nodes
```

No SSH, no torchrun, no environment variables!
