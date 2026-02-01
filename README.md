# Getting started with Distributed Training with Ray

## What is Distributed Training?

Distributed training enables training deep learning models across multiple GPUs or machines by parallelizing computation. The most common approach is **data parallelism**, where:

- Each GPU holds a complete copy of the model
- Training data is split across GPUs (each GPU processes different batches)
- Gradients are synchronized across all GPUs after each backward pass
- Model weights are updated identically on all GPUs

This allows training larger batches and reduces training time proportionally to the number of GPUs.

## PyTorch DistributedDataParallel (DDP)

[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) is PyTorch's native solution for distributed training:

> DDP implements data parallelism at the module level by synchronizing gradients across processes. It uses collective communications from `torch.distributed` to coordinate gradient averaging during the backward pass.

Key characteristics:
- One process per GPU (unlike DataParallel which uses threads)
- Uses NCCL backend for efficient GPU-to-GPU communication
- Requires manual setup of process groups, distributed samplers, and multi-node coordination

## Ray Train

[Ray Train](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) is a library for distributed deep learning that simplifies scaling PyTorch training:

> Ray Train handles the complexity of distributed training setup, including process management, data distribution, and checkpoint handling across workers.

Key benefits:
- Single command to scale from 1 to N GPUs
- Automatic process group initialization
- Built-in fault tolerance and checkpointing
- Seamless integration with Ray's distributed computing ecosystem

## Getting Started with Anyscale

This workshop is designed to run on [Anyscale](https://console.anyscale.com/), a managed Ray platform.

### Creating a Cluster

1. **Sign in to Anyscale Console**
   - Go to [https://console.anyscale.com/](https://console.anyscale.com/)
   - Create an account or sign in

2. **Create a new Workspace**
   - Click "Workspaces" in the left sidebar
   - Click "Create Workspace"
   - Select a compute configuration with GPUs (e.g., `g5.4xlarge` instances)
   - Choose the number of worker nodes based on your GPU requirements

3. **Clone this repository**
   ```bash
   git clone https://github.com/debnsuma/vhol-ray-train.git
   cd vhol-ray-train
   ```

4. **Set up Python environment with `uv`**

   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment
   uv venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate  # Windows
   
   # Install dependencies
   uv pip install torch torchvision ray[train] ipykernel
   
   # For advanced tutorials (03-fsdp-pytorch-ray-deepspeed), also install:
   uv pip install deepspeed matplotlib numpy
   
   # Register Jupyter kernel for IDE integration
   python -m ipykernel install --user --name=ray-train-env --display-name="Python (Ray Train)"
   ```

   **IDE Integration (VSCode, PyCharm, Jupyter)**
   
   After registering the kernel, you can use it in your IDE:
   - **VSCode**: Open a notebook → Click kernel selector (top right) → Select "Python (Ray Train)"
   - **Jupyter Lab/Notebook**: Kernel → Change Kernel → Select "Python (Ray Train)"
   - **PyCharm**: File → Settings → Project → Python Interpreter → Select `.venv/bin/python`

5. **Verify Ray cluster**
   ```bash
   ray status
   ```

## Workshop Structure

```
vhol-ray-train/
├── 01-ddp-pytorch-only/              # Vanilla PyTorch DDP
│   ├── train_ddp.py                  # Training script with manual DDP setup
│   └── launch_multinode_ddp.sh       # Multi-node launch instructions
│
├── 02-ddp-pytorch-ray/               # Ray Train (DDP)
│   ├── Ray_Train_Intro.ipynb         # Comprehensive tutorial notebook (start here)
│   ├── train_ray_ddp.py              # Ray Train with PyTorch DataLoader
│   └── train_ray_ddp_with_ray_data.py # Ray Train with Ray Data
│
└── 03-fsdp-pytorch-ray-deepspeed/    # Advanced: FSDP2 & DeepSpeed
    ├── FSDP2_RayTrain_Tutorial.ipynb # PyTorch FSDP2 with Ray Train
    └── DeepSpeed_RayTrain_Tutorial.ipynb # DeepSpeed ZeRO with Ray Train
```

## Quick Start

### Vanilla PyTorch DDP

```bash
cd 01-ddp-pytorch-only

# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py --epochs 3

# For multi-node, see the launch instructions
./launch_multinode_ddp.sh
```

### Ray Train (DDP)

```bash
cd 02-ddp-pytorch-ray

# Start with the interactive tutorial notebook
jupyter notebook Ray_Train_Intro.ipynb

# Or run the scripts directly
python train_ray_ddp.py --num-workers 8 --epochs 3

# With Ray Data for distributed preprocessing
python train_ray_ddp_with_ray_data.py --num-workers 8 --epochs 3
```

### Advanced: FSDP2 and DeepSpeed

For training large models that don't fit in a single GPU's memory:

```bash
cd 03-fsdp-pytorch-ray-deepspeed

# Start with FSDP2 tutorial (PyTorch native)
jupyter notebook FSDP2_RayTrain_Tutorial.ipynb

# Then try DeepSpeed (Microsoft's ZeRO technology)
jupyter notebook DeepSpeed_RayTrain_Tutorial.ipynb
```

## Comparison

| Aspect | Vanilla PyTorch DDP | Ray Train |
|--------|---------------------|-----------|
| **Launch** | `torchrun` on each node | Single Python command |
| **Process Groups** | Manual init/cleanup | Automatic |
| **Distributed Sampler** | Must create manually | Handled by `prepare_data_loader()` |
| **Multi-node Setup** | SSH, shared storage, coordination | Cluster handles it |
| **Fault Tolerance** | None - any failure stops training | Built-in recovery |
| **Checkpointing** | Manual implementation | Integrated with `ray.train.report()` |

## Further Reading

- [In-Depth Tutorial: Distributed Training from Scratch](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/)
- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Ray Train Getting Started](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [PyTorch DistributedDataParallel Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Anyscale Documentation](https://docs.anyscale.com/)
