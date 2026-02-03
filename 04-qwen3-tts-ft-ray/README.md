# Qwen3-TTS Voice Cloning with Ray Distributed Training

This project demonstrates **voice cloning** using Qwen3-TTS with **distributed training** via Ray Train. It fine-tunes the Qwen3-TTS-12Hz-1.7B-Base model on your audio samples to create a custom voice.

## Overview

**Goal**: Clone a custom voice by fine-tuning Qwen3-TTS on sample audio recordings.

**Pipeline**:
```
Raw Audio Files → Data Processing (Ray Data) → Audio Code Extraction → SFT Training (Ray Train) → Inference
```

**Key Technologies**:
- **Qwen3-TTS**: Open-source TTS model from Alibaba (1.7B parameters)
- **Ray Data**: Distributed data preprocessing
- **Ray Train**: Distributed training orchestration with fault tolerance
- **Supervised Fine-Tuning (SFT)**: Adapts the model to your voice with speaker embedding conditioning

## Fine-Tuning Approach

### What is Supervised Fine-Tuning (SFT)?

SFT adapts a pre-trained model to a specific task by training on labeled examples. For voice cloning:

1. **Input**: Text tokens + Speaker embedding (extracted from your reference audio)
2. **Target**: Audio codes (discrete tokens representing your voice's audio)
3. **Loss**: Cross-entropy on audio code predictions

### How the Training Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. AUDIO CODE EXTRACTION (prepare_data.py)                             │
│     ┌──────────┐      ┌───────────────────────┐      ┌──────────────┐   │
│     │ Your     │ ───► │ Qwen3-TTS-Tokenizer   │ ───► │ Audio Codes  │   │
│     │ Audio    │      │ (12Hz, 16 codebooks)  │      │ [time, 16]   │   │
│     └──────────┘      └───────────────────────┘      └──────────────┘   │
│                                                                          │
│  2. SFT TRAINING (train_qwen_tts.py)                                    │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │                    Qwen3-TTS Model (1.7B params)             │    │
│     │                                                               │    │
│     │   Text Tokens ─────────┐                                      │    │
│     │                        ▼                                      │    │
│     │                  ┌─────────────┐                              │    │
│     │   Speaker Emb ─► │  Transformer │ ───► Predicted Audio Codes  │    │
│     │   (x-vector)     │   Layers    │                              │    │
│     │                  └─────────────┘                              │    │
│     │                                                               │    │
│     │   Loss = CrossEntropy(predicted_codes, target_codes)          │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Speaker Embedding Conditioning

The training uses **speaker embedding conditioning** to teach the model your voice characteristics:

1. **Extract speaker embedding** from reference audio using the model's x-vector extractor
2. **Project the embedding** to match the model's hidden dimension
3. **Add to text embeddings** during forward pass to condition generation on your voice

```python
# Speaker embedding extraction and conditioning
speaker_embedding = model.extract_speaker_embedding(ref_audio, sr=24000)
spk_projected = speaker_projection(speaker_embedding)  # [batch, 1, hidden_dim]
talker_hidden = text_hidden + 0.1 * spk_projected  # Condition on speaker
```

### Audio Codes Explained

The Qwen3-TTS tokenizer converts audio waveforms into discrete tokens:
- **Sample rate**: 12Hz (12 tokens per second of audio)
- **Codebooks**: 16 parallel channels
- **Output shape**: [time_steps, 16]

For a 10-second audio clip: 10 × 12 = 120 time steps × 16 channels = 1,920 tokens

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Per-GPU batch size |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `num_epochs` | 10 | Training epochs (recommend 10-20) |
| `warmup_ratio` | 0.1 | LR warmup percentage |
| `gradient_accumulation` | 4 | Steps before weight update |
| `max_grad_norm` | 1.0 | Gradient clipping |

## Expected Training Time

Training time depends on your data size and hardware:

| Data Size | GPUs | Time per Epoch | Total (10 epochs) |
|-----------|------|----------------|-------------------|
| 50 samples | 4x L40S | ~5-10 min | ~1-2 hours |
| 200 samples | 4x L40S | ~15-20 min | ~3-4 hours |
| 500 samples | 4x L40S | ~30-45 min | ~5-8 hours |

**Why training takes time**:
- Qwen3-TTS has **1.7 billion parameters**
- Each forward pass processes text + audio embeddings
- Backpropagation computes gradients for all trainable parameters

**Tips for faster training**:
- Use more GPUs (Ray Train scales automatically)
- Reduce `num_epochs` (5 may be enough for small datasets)
- Increase `batch_size` if GPU memory allows

## Prerequisites

**Cluster Requirements**:
- Ray cluster with GPU workers (this project uses 4x L40S GPUs)
- Shared storage mounted at `/mnt/cluster_storage`
- Python 3.10+
- ~16GB GPU memory per worker

**Knowledge Prerequisites**:
Complete the previous sections in this repository:
1. `01-ddp-pytorch-only/` - PyTorch DDP fundamentals
2. `02-ddp-pytorch-ray/` - Ray Train basics
3. `03-fsdp-pytorch-ray-deepspeed/` - FSDP and DeepSpeed

## Project Structure

```
04-qwen3-tts-ft-ray/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore patterns
├── configs/
│   └── config.py              # Centralized configuration
├── data/
│   ├── audio1.mp4             # Sample audio files
│   ├── audio2.mp4
│   └── audio3.mp4
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Ray Data processing pipeline
│   ├── prepare_data.py        # Audio code extraction
│   ├── train_qwen_tts.py      # Qwen3-TTS SFT with Ray Train
│   └── inference.py           # Model comparison (base vs fine-tuned)
├── scripts/
│   ├── run_pipeline.py        # End-to-end pipeline runner
│   └── zero_shot_clone.py     # Standalone zero-shot voice cloning
└── output/                    # Generated during execution
    ├── processed/             # Processed audio and JSONL
    ├── checkpoints/           # Training checkpoints
    └── inference/             # Generated audio samples
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Hugging Face Authentication

The Qwen3-TTS models are hosted on Hugging Face and require authentication:

```bash
# Option 1: Login interactively (recommended)
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN="your_huggingface_token"
```

To get your token:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with "read" permissions

### 3. Run the Complete Pipeline

Run each step separately for better control:

```bash
# Step 1: Process audio data (converts MP4/MP3 to WAV, transcribes, segments)
python scripts/run_pipeline.py --process

# Step 2: Extract audio codes (required for training)
python scripts/run_pipeline.py --prepare

# Step 3: Fine-tune Qwen3-TTS (this is the main training step)
python scripts/run_pipeline.py --train --num-workers 4 --num-epochs 10

# Step 4: Compare base vs fine-tuned model
python scripts/run_pipeline.py --infer
```

**Or run everything at once:**
```bash
python scripts/run_pipeline.py --all --num-epochs 10
```

### 4. Output Locations

After running the pipeline:
- **Final Model**: `/mnt/cluster_storage/qwen3-tts-training/final_model/model.pt`
- **Base Model Audio**: `/mnt/cluster_storage/qwen3-tts-training/output/inference/text1_base_model.wav`
- **Fine-tuned Model Audio**: `/mnt/cluster_storage/qwen3-tts-training/output/inference/text1_finetuned_model.wav`

## Zero-Shot Voice Cloning

The project includes a standalone script for **zero-shot voice cloning** - cloning a voice without any fine-tuning, using only a reference audio sample.

### Usage

```bash
# Basic usage (runs on Ray GPU worker)
python scripts/zero_shot_clone.py \
    --ref-audio /path/to/reference_voice.wav \
    --text "Text to synthesize in the cloned voice" \
    --output /path/to/output.wav

# With reference text for ICL mode (better quality)
python scripts/zero_shot_clone.py \
    --ref-audio /path/to/reference_voice.wav \
    --ref-text "Transcript of the reference audio" \
    --text "Text to synthesize" \
    --output /path/to/output.wav

# Run locally without Ray (requires GPU)
python scripts/zero_shot_clone.py \
    --ref-audio /path/to/reference_voice.wav \
    --text "Text to synthesize" \
    --output /path/to/output.wav \
    --local
```

### Voice Cloning Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **X-Vector Only** | Uses speaker embedding from reference audio | Quick cloning, no transcript available |
| **ICL Mode** | Uses reference audio AND transcript | Better voice matching quality |

### Examples

```bash
# Clone a voice with just a sample
python scripts/zero_shot_clone.py \
    --ref-audio samples/my_voice.wav \
    --text "Hello, this is my cloned voice speaking." \
    --output output/cloned_speech.wav

# Better quality with reference transcript
python scripts/zero_shot_clone.py \
    --ref-audio samples/my_voice.wav \
    --ref-text "What I said in the reference audio." \
    --text "Now I can say anything in this voice." \
    --output output/cloned_speech.wav
```

## Pipeline Details

### Step 1: Data Processing with Ray Data

**File**: `src/data_processing.py`

This step processes raw audio files:
1. **Convert** MP4/MP3 to WAV (24kHz, mono)
2. **Transcribe** using Whisper (speech-to-text)
3. **Segment** into training chunks (1-15 seconds)
4. **Save** as JSONL format for training

```python
# Example: Run data processing
from src.data_processing import run_data_processing

jsonl_path = run_data_processing(
    input_dir="data/",
    output_dir="output/processed/"
)
```

**Output Format** (JSONL):
```json
{"audio": "output/processed/wav/audio1_seg0001.wav", "text": "transcribed text", "ref_audio": "output/processed/wav/ref.wav"}
```

### Step 2: Audio Code Extraction

**File**: `src/prepare_data.py`

Extracts audio tokens using Qwen3-TTS-Tokenizer-12Hz:

```python
from src.prepare_data import prepare_training_data

prepare_training_data(
    input_jsonl="output/processed/train.jsonl",
    output_jsonl="output/processed/train_with_codes.jsonl",
    tokenizer_model_path="Qwen/Qwen3-TTS-Tokenizer-12Hz"
)
```

This step runs on GPU workers via Ray and adds `audio_codes` to each sample.

### Step 3: Supervised Fine-Tuning (SFT)

**File**: `src/train_qwen_tts.py`

**This is the core training step** that adapts Qwen3-TTS to your voice using speaker embedding conditioning.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Train Orchestrator                  │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Worker 0  │   Worker 1  │   Worker 2  │   Worker 3          │
│   (GPU 0)   │   (GPU 1)   │   (GPU 2)   │   (GPU 3)           │
├─────────────┴─────────────┴─────────────┴─────────────────────┤
│                    Data Parallel Training                     │
│  - Each worker has full model copy                            │
│  - Gradients synchronized via AllReduce                       │
│  - DistributedSampler for data sharding                       │
└─────────────────────────────────────────────────────────────┘
```

**Training Configuration**:
```python
from src.train_qwen_tts import run_qwen_tts_training, QwenTTSConfig

config = QwenTTSConfig(
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    num_workers=4,           # Use all 4 GPUs
    batch_size=2,            # Per worker
    learning_rate=1e-5,      # Lower LR for stable training
    num_epochs=10,           # 10-20 recommended
    gradient_accumulation_steps=4,
)

checkpoint_path = run_qwen_tts_training(
    train_jsonl="output/processed/train_with_codes.jsonl",
    config=config
)
```

**What happens during training**:
1. Load pre-trained Qwen3-TTS-12Hz-1.7B-Base model
2. Extract speaker embeddings from reference audio
3. For each batch:
   - Tokenize text input
   - Get text embeddings and project them
   - Add speaker embedding conditioning
   - Forward pass with `forward_sub_talker_finetune`
   - Compute cross-entropy loss on audio code predictions
   - Backward pass and gradient update
4. Save checkpoints every epoch

### Step 4: Inference Comparison

**File**: `src/inference.py`

Compare **base Qwen3-TTS model** vs **fine-tuned model** using proper voice cloning:

```python
from src.inference import run_comparison

results = run_comparison(
    test_texts=[
        "Hello, this is a test of the voice cloning system. "
        "I am speaking in a natural and clear voice."
    ],
    ref_audio="output/processed/wav/ref.wav",
    checkpoint_path="final_model/model.pt"
)
```

**Both models now use `generate_voice_clone()`** for proper voice cloning:
- **Base model**: Zero-shot voice cloning with reference audio (x-vector only mode)
- **Fine-tuned model**: Voice cloning with trained speaker representation

**Output**:
- `text1_base_model.wav` - Zero-shot voice clone using base model
- `text1_finetuned_model.wav` - Voice clone using fine-tuned model

## Output Structure

After running the complete pipeline:

```
/mnt/cluster_storage/qwen3-tts-training/
├── output/
│   ├── processed/
│   │   ├── wav/                    # Processed audio segments
│   │   │   ├── audio1_seg0001.wav
│   │   │   └── ...
│   │   ├── train.jsonl             # Training data manifest
│   │   └── train_with_codes.jsonl  # With audio codes for training
│   └── inference/
│       ├── text1_base_model.wav    # Zero-shot voice clone
│       ├── text1_finetuned_model.wav  # Fine-tuned voice clone
│       └── comparison_results.json # Inference metadata
├── final_model/
│   ├── model.pt                    # Final trained model weights
│   └── model_info.json             # Model configuration
└── qwen_tts_finetune/              # Ray Train checkpoints
    └── TorchTrainer_*/
        └── checkpoint_*/
            └── checkpoint.pt
```

## Configuration

All settings are centralized in `configs/config.py`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 2
    learning_rate: float = 1e-5     # Lower LR for stable training
    num_epochs: int = 10            # Recommend 10-20 for good results
    num_workers: int = 4            # Number of GPUs
    use_gpu: bool = True
    storage_path: Path = Path("/mnt/cluster_storage/qwen3-tts-training")
```

## Command Reference

```bash
# Full pipeline
python scripts/run_pipeline.py --all --num-epochs 10

# Data processing only
python scripts/run_pipeline.py --process

# Audio code extraction only
python scripts/run_pipeline.py --prepare

# Training with custom settings
python scripts/run_pipeline.py --train \
    --num-workers 4 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --num-epochs 10

# Inference with checkpoint
python scripts/run_pipeline.py --infer \
    --checkpoint /mnt/cluster_storage/qwen3-tts-training/final_model/model.pt

# Zero-shot voice cloning (standalone)
python scripts/zero_shot_clone.py \
    --ref-audio samples/voice.wav \
    --text "Hello world" \
    --output output.wav

# Run tests
python scripts/run_pipeline.py --test
```

## Testing

Each module includes test functions:

```bash
# Run all tests
python scripts/run_pipeline.py --test

# Or test individual modules
python src/data_processing.py --test
python src/prepare_data.py --test
python src/train_qwen_tts.py --test
python src/inference.py --test
```

## Concepts Covered

| Concept | Where Applied |
|---------|---------------|
| **Ray Data** | `data_processing.py` - Parallel audio processing |
| **Ray Train** | `train_qwen_tts.py` - Distributed training orchestration |
| **SFT** | `train_qwen_tts.py` - Supervised fine-tuning with speaker conditioning |
| **Speaker Embeddings** | `train_qwen_tts.py` - X-vector extraction and conditioning |
| **Audio Codes** | `prepare_data.py` - Discrete audio tokenization |
| **Checkpointing** | `train_qwen_tts.py` - Ray Checkpoint API |
| **Voice Cloning** | `inference.py`, `zero_shot_clone.py` - Zero-shot and fine-tuned |

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python scripts/run_pipeline.py --train --batch-size 1
```

Or increase gradient accumulation:
```python
# In train_qwen_tts.py
QwenTTSConfig(gradient_accumulation_steps=8)
```

### Training Loss Not Decreasing

- Use a lower learning rate (1e-5 or 5e-6)
- Ensure speaker embedding conditioning is working
- Check that audio codes are properly extracted

### Training Too Slow

Check if all GPUs are being used:
```bash
ray status
nvidia-smi
```

### Voice Doesn't Sound Different

- Train for more epochs (10-20 recommended)
- Ensure you have enough training samples (50+ recommended)
- Check that audio codes were properly extracted
- Verify speaker embedding is being used in training

### Ray Cluster Issues

Check cluster status:
```bash
ray status
```

Ensure shared storage is accessible:
```bash
ls /mnt/cluster_storage
```

### Model Loading Errors

Install Qwen3-TTS package:
```bash
pip install qwen-tts
```

### File Not Found on Worker Node

Files on the head node's local filesystem (e.g., `/home/ray/...`) are not accessible from worker nodes. Solutions:
- Use shared storage (`/mnt/cluster_storage/`)
- Load files on head node and pass data to workers (used in `zero_shot_clone.py`)

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Fine-tuning Guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [ComfyUI-FL-Qwen3TTS SFT Reference](https://github.com/filliptm/ComfyUI-FL-Qwen3TTS/tree/main/src/qwen_tts/finetuning)

## Next Steps

After completing this project:
1. Try fine-tuning with more audio data (100+ samples)
2. Experiment with different learning rates and epochs
3. Deploy the fine-tuned model with Ray Serve
4. Explore multi-speaker fine-tuning
