"""
Ray Train with Ray Data

This script demonstrates distributed training using Ray Train with Ray Data
for distributed data loading and preprocessing.

Benefits of Ray Data:
  - Preprocessing runs on CPU workers, not GPU workers
  - Data streams to GPU workers on-demand (pipelined execution)
  - Works with any data source (Parquet, S3, images, etc.)

Run:
    python train_ray_ddp_with_ray_data.py --num-workers 2 --epochs 3
"""

import os
import tempfile
import argparse
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from torchvision.datasets import MNIST

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


def build_model():
    """Build ResNet18 modified for MNIST (1 channel input)."""
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def transform_batch(batch):
    """Preprocessing function executed on Ray Data workers (CPU)."""
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    images = [
        transform(Image.fromarray(np.array(img, dtype=np.uint8))).numpy()
        for img in batch["image"]
    ]
    return {"image": np.stack(images), "label": np.array(batch["label"])}


def train_func(config):
    """Training function executed on each GPU worker."""
    # Ray Train API: prepare_model handles DDP wrapping
    model = build_model()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    device = ray.train.torch.get_device()

    for epoch in range(config["epochs"]):
        # Get this worker's data shard from Ray Data
        train_shard = ray.train.get_dataset_shard("train")

        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_shard.iter_torch_batches(batch_size=config["batch_size"]):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Ray Train API: report metrics and checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.module.state_dict(), os.path.join(tmpdir, "model.pt"))
            ray.train.report(
                {"loss": total_loss / num_batches, "epoch": epoch + 1},
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

    # Create Ray Dataset from MNIST
    mnist = MNIST(root="./data", train=True, download=True)
    df = pd.DataFrame({
        "image": mnist.data.numpy().tolist(),
        "label": mnist.targets.numpy(),
    })
    train_ds = ray.data.from_pandas(df).map_batches(transform_batch)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True),
        run_config=RunConfig(storage_path="/mnt/cluster_storage"),
        datasets={"train": train_ds},
    )

    result = trainer.fit()
    print(f"Training complete! Final loss: {result.metrics['loss']:.4f}")

    # Cleanup
    cleanup()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
