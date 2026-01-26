# Getting Started with Distributed Training

Hands-on webinar workshop: Vanilla PyTorch DDP vs Ray Train

## Workshop Structure

```
vhol-ray-train/
├── 01-ddp-pytorch-only/          # The hard way
│   ├── train_ddp.py              # Training script
│   ├── launch_multinode_ddp.sh   # Launcher
│   └── run_multinode_ddp.py      # Orchestration
└── 02-ddp-pytorch-ray/           # The easy way
    ├── version1-without-ray-data/
    │   └── train_ray_ddp.py
    └── version2-with-ray-data/
        └── train_ray_ddp_ray_data.py
```

## Demo Commands

### Vanilla DDP (needs orchestration)

```bash
cd 01-ddp-pytorch-only
./launch_multinode_ddp.sh 3 128 0.001
```

### Ray Train (just works)

```bash
cd 02-ddp-pytorch-ray/version1-without-ray-data
python train_ray_ddp.py --num-workers 8 --epochs 3
```

## Key Comparison

| Aspect | Vanilla DDP | Ray Train |
|--------|-------------|-----------|
| Launch | Needs orchestration script | Single command |
| Code | ~100 lines + launcher | ~90 lines total |
| Setup | 6+ manual steps | 3 simple changes |
| Fault tolerance | None | Built-in |
| Scaling | Modify launcher | Change `--num-workers` |

## Cluster Info

```bash
ray status  # Check cluster resources
```
