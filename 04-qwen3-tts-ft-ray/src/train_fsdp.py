"""
Distributed Training with FSDP and Ray Train

This module implements distributed training using:
- PyTorch FSDP2 for model sharding
- Ray Train for orchestration and fault tolerance
- Mixed precision training for efficiency

Learning Objectives:
- Distributed training with FSDP2 + Ray Train
- Memory optimization techniques
- Checkpointing strategies
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import TrainConfig, ModelConfig, PROCESSED_DIR, RAY_STORAGE_PATH


# ============================================================================
# Training Function (Self-contained for Ray serialization)
# ============================================================================

def get_train_func():
    """
    Returns the training function with all dependencies embedded.
    This is necessary for Ray to properly serialize the function.
    """

    def train_func(config: Dict[str, Any]):
        """
        Training function executed on each Ray worker.
        All imports and class definitions are inside to ensure proper serialization.
        """
        import os
        import json
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler
        from typing import Dict, Any, Optional, List
        from ray import train

        # ================================================================
        # Model Definition (embedded)
        # ================================================================
        class SimpleTTSModel(nn.Module):
            """Simple TTS model for demonstrating FSDP training patterns."""

            def __init__(
                self,
                vocab_size: int = 32000,
                hidden_dim: int = 512,
                num_heads: int = 8,
                num_layers: int = 6,
                audio_channels: int = 1,
                max_seq_len: int = 2048,
            ):
                super().__init__()
                self.hidden_dim = hidden_dim

                # Audio encoder
                self.audio_conv = nn.Sequential(
                    nn.Conv1d(audio_channels, hidden_dim // 4, kernel_size=10, stride=5),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2),
                    nn.GELU(),
                )

                # Transformer layers
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim * 4,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(num_layers)
                ])

                self.output_proj = nn.Linear(hidden_dim, hidden_dim)
                self.criterion = nn.MSELoss()

            def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                encoded = self.audio_conv(audio)
                return encoded.transpose(1, 2)

            def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
                audio_enc = self.encode_audio(audio)
                hidden = audio_enc
                for layer in self.layers:
                    hidden = layer(hidden)
                output = self.output_proj(hidden)
                loss = self.criterion(output, audio_enc.detach())
                return {"loss": loss, "output": output}

        # ================================================================
        # Dataset Definition (embedded)
        # ================================================================
        class TTSDataset(torch.utils.data.Dataset):
            def __init__(self, jsonl_path: str, max_samples: Optional[int] = None,
                         max_audio_len: int = 160000):
                import soundfile as sf
                self.data = []
                self.max_audio_len = max_audio_len

                with open(jsonl_path, 'r') as f:
                    for i, line in enumerate(f):
                        if max_samples and i >= max_samples:
                            break
                        item = json.loads(line)
                        if os.path.exists(item["audio"]):
                            self.data.append(item)

                print(f"Loaded {len(self.data)} samples")

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                import soundfile as sf
                import numpy as np
                item = self.data[idx]
                audio, sr = sf.read(item["audio"])
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if len(audio) > self.max_audio_len:
                    audio = audio[:self.max_audio_len]
                else:
                    audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
                return {"audio": torch.tensor(audio, dtype=torch.float32), "text": item["text"]}

        def collate_fn(batch):
            return {
                "audio": torch.stack([item["audio"] for item in batch]),
                "text": [item["text"] for item in batch],
            }

        # ================================================================
        # FSDP2 Setup (embedded)
        # ================================================================
        def setup_fsdp2(model, world_size):
            from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

            for layer in model.layers:
                fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

            fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
            return model

        # ================================================================
        # Main Training Logic
        # ================================================================
        rank = train.get_context().get_world_rank()
        local_rank = train.get_context().get_local_rank()
        world_size = train.get_context().get_world_size()

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        print(f"Worker {rank}/{world_size} starting on {device}")

        # Load dataset
        dataset = TTSDataset(config["train_jsonl"], max_samples=config.get("max_samples"))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], sampler=sampler,
            collate_fn=collate_fn, num_workers=2, pin_memory=True
        )

        # Create model
        print("Creating SimpleTTSModel...")
        model = SimpleTTSModel(
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
        )
        model = model.to(device)

        # Apply FSDP2
        if world_size > 1 and config.get("use_fsdp", True):
            print("Applying FSDP2 sharding...")
            model = setup_fsdp2(model, world_size)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Scheduler
        total_steps = len(dataloader) * config["num_epochs"]
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.1, 1 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        model.train()
        global_step = 0

        for epoch in range(config["num_epochs"]):
            sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                audio = batch["audio"].to(device)
                outputs = model(audio)
                loss = outputs["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if batch_idx % 20 == 0 and rank == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

            avg_loss = epoch_loss / max(num_batches, 1)
            metrics = {"loss": avg_loss, "epoch": epoch + 1, "learning_rate": scheduler.get_last_lr()[0]}

            # Checkpoint - save FULL state dict (not sharded)
            if (epoch + 1) % config.get("save_every_n_epochs", 1) == 0:
                checkpoint_dir = f"/tmp/checkpoint-epoch-{epoch}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                # For FSDP models, we need to get the full state dict
                # Use get_model_state_dict with full_state_dict option
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

                # Check if model is FSDP-wrapped
                if world_size > 1 and config.get("use_fsdp", True):
                    # Get full state dict on rank 0
                    full_state_dict = get_model_state_dict(
                        model,
                        options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                    )
                else:
                    full_state_dict = model.state_dict()

                if rank == 0:
                    torch.save({
                        "model_state_dict": full_state_dict,
                        "optimizer_state_dict": None,  # Skip optimizer state for simplicity
                        "epoch": epoch,
                        "loss": avg_loss,
                    }, f"{checkpoint_dir}/checkpoint.pt")

                if dist.is_initialized():
                    dist.barrier()

                checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
                train.report(metrics, checkpoint=checkpoint)
            else:
                train.report(metrics)

            if rank == 0:
                print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        print("Training complete!")

    return train_func


# ============================================================================
# Main Training Function
# ============================================================================

def run_training(
    train_jsonl: Optional[str] = None,
    config: Optional[TrainConfig] = None,
) -> Optional[str]:
    """Run distributed training with Ray Train and FSDP."""
    if config is None:
        config = TrainConfig()

    train_jsonl = train_jsonl or str(PROCESSED_DIR / "train.jsonl")

    train_config = {
        "train_jsonl": train_jsonl,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "use_fsdp": True,
        "cpu_offload": config.cpu_offload,
        "save_every_n_epochs": config.save_every_n_epochs,
        "hidden_dim": 512,
        "num_layers": 6,
        "max_samples": 200,  # Limit for demo
    }

    scaling_config = ScalingConfig(
        num_workers=config.num_workers,
        use_gpu=config.use_gpu,
        resources_per_worker={"CPU": 4, "GPU": 1}
    )

    run_config = RunConfig(
        name="tts_fsdp_training",
        storage_path=str(config.storage_path),
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    )

    # Get the self-contained training function
    train_func = get_train_func()

    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print("=" * 60)
    print("Starting TTS Training with Ray Train + FSDP")
    print("=" * 60)
    print(f"Workers: {config.num_workers}")
    print(f"Batch size per worker: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Storage: {config.storage_path}")
    print("=" * 60)

    result = trainer.fit()

    print("\nTraining complete!")

    # Save final model to well-known location
    final_model_dir = config.storage_path / "final_model"

    if result.checkpoint:
        checkpoint_path = str(result.checkpoint.path)
        print(f"Best checkpoint: {checkpoint_path}")

        # Copy the best checkpoint to final_model directory
        import shutil
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint file
        src_checkpoint = Path(checkpoint_path) / "checkpoint.pt"
        dst_checkpoint = final_model_dir / "model.pt"

        if src_checkpoint.exists():
            shutil.copy2(src_checkpoint, dst_checkpoint)
            print(f"\n{'='*60}")
            print("FINAL MODEL SAVED!")
            print(f"{'='*60}")
            print(f"Location: {dst_checkpoint}")
            print(f"{'='*60}\n")

            # Also save model info
            import json
            model_info = {
                "checkpoint_source": checkpoint_path,
                "model_type": "SimpleTTSModel",
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "training_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
            }
            with open(final_model_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)

        return str(final_model_dir / "model.pt")

    return None


# ============================================================================
# Testing Function
# ============================================================================

def test_training_components():
    """Test training components locally"""
    print("Testing training components...")

    # Test model creation
    print("\n1. Testing model creation...")

    class SimpleTTSModel(nn.Module):
        def __init__(self, hidden_dim=256, num_layers=2):
            super().__init__()
            self.audio_conv = nn.Sequential(
                nn.Conv1d(1, hidden_dim // 4, kernel_size=10, stride=5),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 4, hidden_dim, kernel_size=5, stride=2),
            )
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            self.criterion = nn.MSELoss()

        def forward(self, audio):
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            enc = self.audio_conv(audio).transpose(1, 2)
            for layer in self.layers:
                enc = layer(enc)
            out = self.output_proj(enc)
            return {"loss": self.criterion(out, enc.detach())}

    model = SimpleTTSModel()
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    audio = torch.randn(2, 16000)
    output = model(audio)
    assert "loss" in output, "Should have loss"
    print(f"   Loss: {output['loss'].item():.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TTS model with FSDP + Ray")
    parser.add_argument("--train-jsonl", type=str, default=str(PROCESSED_DIR / "train.jsonl"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.test:
        test_training_components()
    else:
        config = TrainConfig(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        )
        run_training(train_jsonl=args.train_jsonl, config=config)
