"""
Vanilla PyTorch DDP Training Script

This shows all the MANUAL steps required for distributed training without Ray Train.
Compare with the Ray Train version to see how much simpler it becomes!
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
import argparse


# =============================================================================
# STEP 1: Manual distributed setup (Ray Train does this automatically)
# =============================================================================
def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


# =============================================================================
# STEP 2: Build model (same for both vanilla and Ray Train)
# =============================================================================
def build_model():
    model = resnet18(num_classes=10)
    # Modify for MNIST (1 channel instead of 3)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


# =============================================================================
# STEP 3: Manual data loading with DistributedSampler
# (Ray Train: just use prepare_data_loader())
# =============================================================================
def get_dataloader(batch_size):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Download only on rank 0
    if rank == 0:
        datasets.MNIST(root="./data", train=True, download=True)
    dist.barrier()

    dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)

    # MUST use DistributedSampler - Ray Train handles this automatically
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler), sampler


# =============================================================================
# STEP 4: Training loop
# =============================================================================
def train(epochs, batch_size, lr):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Get data - MUST manually create DistributedSampler
    train_loader, sampler = get_dataloader(batch_size)

    # Create model and wrap with DDP - Ray Train: just use prepare_model()
    model = build_model().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # MUST call set_epoch for proper shuffling - easy to forget!
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(local_rank), labels.cuda(local_rank)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Save checkpoint (only rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), "model_ddp.pt")
        print("Model saved to model_ddp.pt")


# =============================================================================
# MAIN - Must be launched with torchrun
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    setup()

    if dist.get_rank() == 0:
        print(f"Training with {dist.get_world_size()} GPUs")

    train(args.epochs, args.batch_size, args.lr)

    cleanup()


if __name__ == "__main__":
    main()
