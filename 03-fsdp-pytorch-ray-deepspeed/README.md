# Advanced Distributed Training: FSDP2 and DeepSpeed with Ray Train

This folder contains comprehensive tutorials on advanced distributed training techniques using PyTorch FSDP2 and Microsoft DeepSpeed with Ray Train. These tutorials demonstrate how to train large models that don't fit in a single GPU's memory by sharding model parameters, gradients, and optimizer states across multiple GPUs and nodes.

## Overview

### What You'll Learn

- **FSDP2 (Fully Sharded Data Parallel)**: PyTorch's native solution for model sharding
- **DeepSpeed ZeRO**: Microsoft's memory optimization technology
- **Memory optimization techniques**: CPU offloading, mixed precision, gradient checkpointing
- **Distributed checkpointing**: Save and load sharded models efficiently
- **Production-ready patterns**: Best practices for large-scale training

### Tutorials

| Notebook | Description |
|----------|-------------|
| `FSDP2_RayTrain_Tutorial.ipynb` | Complete guide to PyTorch FSDP2 with Ray Train |
| `DeepSpeed_RayTrain_Tutorial.ipynb` | DeepSpeed ZeRO integration with Ray Train |

## Flow

1. Start with `FSDP2_RayTrain_Tutorial.ipynb` to learn PyTorch's native FSDP2 approach
2. Continue with `DeepSpeed_RayTrain_Tutorial.ipynb` to learn DeepSpeed's ZeRO technology
3. Compare both approaches to understand when to use each

## Environment Setup

**Note**: Environment setup should be completed at the repository root level. See the main [README.md](../README.md) for instructions on setting up the virtual environment and installing dependencies from `requirements.txt`.

If you've already set up the environment at the root, you can use the same virtual environment for these tutorials. Simply activate it and select the "Python (Ray Train)" kernel in your IDE.

```python
# Run this in a Python shell or notebook cell
import torch
import ray
import deepspeed

print(f"PyTorch version: {torch.__version__}")
print(f"Ray version: {ray.__version__}")
print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

## Quick Start

### 1. Start with FSDP2

The FSDP2 tutorial is recommended as the first tutorial because:
- It uses PyTorch's native APIs (no external dependencies beyond Ray)
- The concepts transfer directly to DeepSpeed
- It demonstrates PyTorch's Distributed Checkpoint (DCP) API

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Start Jupyter
jupyter notebook FSDP2_RayTrain_Tutorial.ipynb
```

**Key Learning Points:**
- How FSDP2 shards model parameters across GPUs
- Configuring CPU offloading and mixed precision
- Using PyTorch Distributed Checkpoint (DCP) for sharded checkpoints
- Memory profiling with PyTorch's memory snapshot API

### 2. Continue with DeepSpeed

After completing FSDP2, the DeepSpeed tutorial shows:
- How to achieve similar memory savings with a different API
- Configuration-based approach vs. programmatic API
- When to choose DeepSpeed over FSDP2

```bash
jupyter notebook DeepSpeed_RayTrain_Tutorial.ipynb
```

**Key Learning Points:**
- DeepSpeed ZeRO stages (1, 2, 3, Infinity)
- Configuration-driven training setup
- Built-in checkpointing methods
- Comparison with FSDP2

## Architecture Overview

### FSDP2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FSDP2 Training Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Worker 0          Worker 1          Worker 2          Worker 3 │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐ │
│  │ Model   │      │ Model   │      │ Model   │      │ Model   │ │
│  │ Shard 0 │      │ Shard 1 │      │ Shard 2 │      │ Shard 3 │ │
│  │         │      │         │      │         │      │         │ │
│  │ Forward │ ────│ Forward │ ────│ Forward │ ────│ Forward │ │
│  │         │      │         │      │         │      │         │ │
│  │ All-    │      │ All-    │      │ All-    │      │ All-    │ │
│  │ Gather  │◄─────┤ Gather  │◄─────┤ Gather  │◄─────┤ Gather  │ │
│  │         │      │         │      │         │      │         │ │
│  │ Backward│      │ Backward│      │ Backward│      │ Backward│ │
│  │         │      │         │      │         │      │         │ │
│  │ Reduce  │ ────│ Reduce  │ ────│ Reduce  │ ────│ Reduce  │ │
│  │ Scatter │      │ Scatter │      │ Scatter │      │ Scatter │ │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Sharding**: Each worker holds a fraction of model parameters
- **All-Gather**: Collect full parameters for forward/backward pass
- **Reduce-Scatter**: Distribute gradients and update sharded parameters
- **Resharding**: Option to free gathered parameters after forward pass

### DeepSpeed ZeRO Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DeepSpeed ZeRO Stages                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ZeRO-1: Partition Optimizer States                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Params  │  │ Params  │  │ Params  │  │ Params  │      │
│  │ Grads   │  │ Grads   │  │ Grads   │  │ Grads   │      │
│  │ Opt[0]  │  │ Opt[1]  │  │ Opt[2]  │  │ Opt[3]  │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                                              │
│  ZeRO-2: Partition Optimizer States + Gradients             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Params  │  │ Params  │  │ Params  │  │ Params  │      │
│  │ Grad[0] │  │ Grad[1] │  │ Grad[2] │  │ Grad[3] │      │
│  │ Opt[0]  │  │ Opt[1]  │  │ Opt[2]  │  │ Opt[3]  │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                                              │
│  ZeRO-3: Partition Everything                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │Param[0] │  │Param[1] │  │Param[2] │  │Param[3] │      │
│  │Grad[0]  │  │Grad[1]  │  │Grad[2]  │  │Grad[3]  │      │
│  │ Opt[0]  │  │ Opt[1]  │  │ Opt[2]  │  │ Opt[3]  │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Memory Optimization Techniques

### CPU Offloading

**What it does:**
- Stores sharded parameters, gradients, and optimizer states on CPU
- Copies to GPU only when needed for computation
- Reduces peak GPU memory usage

**When to use:**
- Model is too large for GPU memory even with sharding
- You have sufficient CPU memory
- Training speed is acceptable with CPU-GPU transfers

**Trade-offs:**
- Increased CPU-GPU data transfer overhead
- Slower training compared to GPU-only
- Requires sufficient CPU memory

### Mixed Precision Training

**What it does:**
- Uses FP16 (or BF16) for parameters and activations
- Maintains FP32 for critical operations (loss computation, etc.)
- Automatic loss scaling to prevent underflow

**Benefits:**
- ~2x memory reduction for activations
- Faster computation on tensor cores (V100, A100, etc.)
- Minimal accuracy impact with proper scaling

**When to use:**
- GPU supports tensor cores (Volta architecture and newer)
- Model can tolerate FP16 precision
- Memory is a constraint

### Resharding After Forward Pass

**With resharding (`reshard_after_forward=True`):**
- Lower peak memory during forward pass
- Weights are re-gathered during backward pass
- More communication overhead
- Better for memory-constrained scenarios

**Without resharding (`reshard_after_forward=False`):**
- Weights stay gathered after forward pass
- Higher peak memory
- Less communication overhead
- Better for communication-constrained scenarios

## Checkpointing Strategies

### FSDP2: PyTorch Distributed Checkpoint (DCP)

```python
import torch.distributed.checkpoint as dcp

# Save sharded checkpoint
dcp.save(
    state_dict={"app": AppState(model, optimizer, epoch)},
    checkpoint_id=checkpoint_dir
)

# Load sharded checkpoint (automatic resharding)
dcp.load(
    state_dict={"app": app_state},
    checkpoint_id=checkpoint_dir
)
```

**Advantages:**
- Native PyTorch API
- Automatic resharding if worker count changes
- Efficient parallel I/O
- Supports custom state objects via `Stateful` protocol

### DeepSpeed: Built-in Checkpointing

```python
# Save checkpoint
model_engine.save_checkpoint(
    save_dir=checkpoint_dir,
    tag=f"epoch_{epoch}",
    client_state={"epoch": epoch}
)

# Load checkpoint
_, client_state = model_engine.load_checkpoint(
    load_dir=checkpoint_dir,
    tag=latest_tag
)
```

**Advantages:**
- Simpler API (no wrapper classes needed)
- Automatic optimizer and scheduler state handling
- Built-in support for ZeRO stages
- Widely used in production LLM training

## Comparison: FSDP2 vs DeepSpeed

| Aspect | FSDP2 | DeepSpeed ZeRO |
|--------|-------|----------------|
| **Origin** | PyTorch native | Microsoft Research |
| **API Style** | Programmatic (Python) | Configuration-based (JSON/dict) |
| **Ecosystem** | PyTorch ecosystem | Framework-agnostic (HuggingFace, Lightning, etc.) |
| **Setup Complexity** | Medium | Low (config-driven) |
| **CPU Offloading** | Basic support | Advanced (ZeRO-Infinity with NVMe) |
| **Optimizer Fusion** | Limited | Built-in (FusedAdam, etc.) |
| **Checkpointing** | PyTorch DCP | Built-in methods |
| **Adoption** | Growing | Widely used (LLM training) |
| **Best For** | PyTorch-native workflows | Large language models, HuggingFace |

### When to Choose FSDP2

- You want native PyTorch integration
- You're using PyTorch's ecosystem (DTensor, etc.)
- You prefer Python API over config files
- You need fine-grained control over sharding
- You're building custom training loops

### When to Choose DeepSpeed

- Training very large models (billions of parameters)
- You need advanced CPU/NVMe offloading (ZeRO-Infinity)
- Using HuggingFace Transformers (excellent integration)
- You want battle-tested LLM training recipes
- You need fused optimizers for performance
- You prefer configuration-driven setup

## Performance Considerations

### Communication Overhead

Both FSDP2 and DeepSpeed add communication overhead compared to DDP:

- **All-Gather operations**: Collect full parameters for forward pass
- **Reduce-Scatter operations**: Distribute gradients and updates
- **Bandwidth requirements**: Higher than DDP due to parameter communication

**Mitigation strategies:**
- Use gradient accumulation to reduce communication frequency
- Enable communication/computation overlap when available
- Use appropriate bucket sizes for gradient reduction
- Consider tensor parallelism for very large models

### Memory Profiling

Both tutorials include GPU memory profiling using PyTorch's memory snapshot API:

```python
# Enable memory profiling
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... training code ...

# Save snapshot
torch.cuda.memory._dump_snapshot(snapshot_path)

# Visualize using PyTorch's memory visualizer
# python -m torch.utils.show_memory_snapshot snapshot.pickle
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Errors**

- **Solution**: Enable CPU offloading or reduce batch size
- **FSDP2**: Set `offload_policy=CPUOffloadPolicy()`
- **DeepSpeed**: Enable `offload_optimizer` and `offload_param` in config

**2. Slow Training**

- **Check**: Communication overhead vs. memory savings
- **Solution**: Try `reshard_after_forward=False` (FSDP2) or ZeRO-2 instead of ZeRO-3 (DeepSpeed)
- **Consider**: Using gradient accumulation to reduce communication frequency

**3. Checkpoint Loading Fails**

- **FSDP2**: Ensure all workers can access checkpoint directory
- **DeepSpeed**: Verify checkpoint tag exists and is accessible
- **Both**: Check that worker count matches (or use automatic resharding)

**4. CUDA Errors**

- **Check**: CUDA version compatibility with PyTorch and DeepSpeed
- **Solution**: Ensure all workers have compatible CUDA versions
- **Verify**: `torch.cuda.is_available()` returns `True` on all workers

### Getting Help

- **Ray Train Documentation**: https://docs.ray.io/en/latest/train/
- **FSDP2 Tutorial**: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **DeepSpeed Documentation**: https://www.deepspeed.ai/
- **Ray Community**: https://discuss.ray.io/

## Next Steps

After completing these tutorials:

1. **Scale up**: Try training with more workers or larger models
2. **Hybrid parallelism**: Combine FSDP2/DeepSpeed with tensor parallelism
3. **Production deployment**: Use cloud storage (S3, GCS) for checkpoints
4. **Hyperparameter tuning**: Integrate with Ray Tune
5. **Real-world models**: Apply to your own large models

## Additional Resources

### Documentation

- [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch Distributed Checkpoint](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)

### Examples and Templates

- [Ray Train FSDP2 Example](https://docs.ray.io/en/latest/train/examples/pytorch/fsdp2.html)
- [Ray Train DeepSpeed Example](https://docs.ray.io/en/latest/train/examples/deepspeed/deepspeed_example.html)
- [Anyscale LLM Finetuning Template](https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed)

### Blog Posts and Case Studies

- [Canva's Stable Diffusion Training with Ray](https://www.anyscale.com/blog/scalable-and-cost-efficient-stable-diffusion-pre-training-with-ray)
- [Fast Data Loading for ML Training with Ray Data](https://www.anyscale.com/blog/fast-flexible-scalable-data-loading-for-ml-training-with-ray-data)

## File Structure

```
03-fsdp-pytorch-ray-deepspeed/
├── README.md                          # This file
├── FSDP2_RayTrain_Tutorial.ipynb     # FSDP2 tutorial notebook
├── DeepSpeed_RayTrain_Tutorial.ipynb # DeepSpeed tutorial notebook
├── images/                            # Memory profiling images
│   ├── all_strategies_profile.png
│   ├── cpu_offload_profile.png
│   ├── gpu_memory_profile.png
│   ├── mixed_precision_profile.png
│   └── reshard_after_forward_memory_profile.png
└── .venv/                            # Virtual environment (created by user)
```

