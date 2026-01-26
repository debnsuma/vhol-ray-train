"""
Ray Train DDP - Compare this to the vanilla PyTorch DDP version!

Only 3 changes needed:
  1. prepare_data_loader() - handles DistributedSampler automatically
  2. prepare_model() - handles DDP wrapping automatically
  3. ray.train.report() - handles metrics/checkpoints automatically
"""

import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
import argparse

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


def build_model():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def train_func(config):
    # Standard PyTorch data loading - NO DistributedSampler needed!
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # [1] prepare_data_loader - handles distributed sampling automatically
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # [2] prepare_model - handles DDP wrapping automatically
    model = build_model()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        # No sampler.set_epoch() needed - Ray handles it!
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # [3] ray.train.report - handles metrics and checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.module.state_dict(), os.path.join(tmpdir, "model.pt"))
            ray.train.report(
                {"loss": total_loss / len(train_loader), "epoch": epoch + 1},
                checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
            )


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
        train_loop_config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True),
        run_config=RunConfig(storage_path="/mnt/cluster_storage"),
    )

    result = trainer.fit()
    print(f"Training complete! Final loss: {result.metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
