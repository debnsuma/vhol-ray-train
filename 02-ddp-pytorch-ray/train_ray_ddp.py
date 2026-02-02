"""
Ray Train DDP Training

This script demonstrates distributed training using Ray Train with PyTorch.
Ray Train handles all the distributed training complexity automatically.

Key Ray Train APIs:
  - prepare_data_loader(): Adds DistributedSampler and handles device placement
  - prepare_model(): Wraps model with DDP and moves to correct device
  - ray.train.report(): Reports metrics and checkpoints

Run:
    python train_ray_ddp.py --num-workers 2 --epochs 3
"""

import os
import tempfile
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


def build_model():
    """Build ResNet18 modified for MNIST (1 channel input)."""
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def train_func(config):
    """Training function executed on each worker."""
    # Standard PyTorch data loading
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Ray Train API: prepare_data_loader handles DistributedSampler
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Ray Train API: prepare_model handles DDP wrapping
    model = build_model()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Ray Train API: report metrics and checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.module.state_dict(), os.path.join(tmpdir, "model.pt"))
            ray.train.report(
                {"loss": total_loss / len(train_loader), "epoch": epoch + 1},
                checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
            )


def cleanup():
    """Clean up GPU and CPU memory from all workers."""
    import gc

    # Clear Python garbage
    gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ray.init()

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True),
        run_config=RunConfig(storage_path="/mnt/cluster_storage"),
    )

    result = trainer.fit()
    print(f"Training complete! Final loss: {result.metrics['loss']:.4f}")

    # Cleanup
    cleanup()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
