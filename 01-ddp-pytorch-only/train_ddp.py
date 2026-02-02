"""
PyTorch Distributed Data Parallel (DDP) Training

This script demonstrates DDP - PyTorch's approach to distributed training where
each process maintains its own copy of the model and gradients are synchronized
across all processes during the backward pass.

Run with torchrun:
    torchrun --nproc_per_node=4 train_ddp.py --epochs 3

For multi-node training, see launch_multinode_ddp.sh
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18


def setup():
    """Initialize the distributed process group.

    This creates a communication group for all processes to coordinate
    gradient synchronization. The NCCL backend is optimized for GPU training.
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    """Clean up the distributed process group and free memory."""
    import gc

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clear Python garbage
    gc.collect()

    # Destroy process group
    dist.destroy_process_group()


def get_dataloader(batch_size):
    """Create a distributed dataloader.

    DistributedSampler partitions the dataset across all processes,
    ensuring each GPU sees a unique subset of the data.
    """
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # Download data only on rank 0, then synchronize
    if dist.get_rank() == 0:
        datasets.MNIST(root="./data", train=True, download=True)
    dist.barrier()

    dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler), sampler


def train(epochs, batch_size, lr):
    """Main training loop with DDP."""
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Create dataloader with distributed sampler
    train_loader, sampler = get_dataloader(batch_size)

    # Build model and wrap with DDP
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # IMPORTANT: Set epoch for proper shuffling across epochs
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(local_rank), labels.cuda(local_rank)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()  # Gradients are synchronized automatically
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "model_ddp.pt")
        print("Model saved to model_ddp.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Step1: Setup for Distributed Training 
    setup()

    # Step2: Start the training 
    train(args.epochs, args.batch_size, args.lr)

    # Step3: Destroy the setup 
    cleanup()
