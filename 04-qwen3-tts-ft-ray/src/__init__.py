"""
Qwen3-TTS Voice Cloning with Ray Distributed Training

This package provides:
- data_processing: Audio preprocessing with Ray Data
- prepare_data: Tokenization for Qwen3-TTS training
- train_fsdp: Distributed training with FSDP + Ray Train
- inference: Model comparison (base vs fine-tuned)
"""

__version__ = "0.1.0"
