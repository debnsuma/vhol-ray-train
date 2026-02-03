#!/usr/bin/env python3
"""
End-to-End Pipeline Runner

This script orchestrates the complete voice cloning pipeline:
1. Data Processing (with Ray Data)
2. Data Preparation (tokenization)
3. Distributed Training (with FSDP + Ray Train)
4. Inference Comparison (base vs fine-tuned)

Usage:
    python scripts/run_pipeline.py --all           # Run full pipeline
    python scripts/run_pipeline.py --process       # Data processing only
    python scripts/run_pipeline.py --train         # Training only
    python scripts/run_pipeline.py --infer         # Inference only
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs.config import (
    DataConfig, ModelConfig, TrainConfig, InferenceConfig,
    setup_directories, PROCESSED_DIR, OUTPUT_DIR, RAY_STORAGE_PATH
)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step_1_process_data(args):
    """Step 1: Process raw audio data with Ray Data"""
    print_header("STEP 1: Data Processing with Ray Data")

    from data_processing import run_data_processing

    config = DataConfig()

    # Override with args if provided
    input_dir = args.input_dir or str(config.raw_audio_dir)
    output_dir = args.output_dir or str(config.processed_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    jsonl_path = run_data_processing(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config
    )

    print(f"\n✓ Data processing complete!")
    print(f"  Output: {jsonl_path}")

    return jsonl_path


def step_2_prepare_data(args, train_jsonl: str = None):
    """Step 2: Prepare data for training (extract audio codes)"""
    print_header("STEP 2: Audio Code Extraction (Qwen3-TTS Tokenizer)")

    from prepare_data import prepare_training_data, prepare_data_simple

    train_jsonl = train_jsonl or str(PROCESSED_DIR / "train.jsonl")
    output_jsonl = str(PROCESSED_DIR / "train_with_codes.jsonl")

    if args.skip_tokenization:
        print("Skipping tokenization (using simple copy)...")
        output_path = prepare_data_simple(train_jsonl, output_jsonl)
    else:
        print("Extracting audio codes with Qwen3-TTS-Tokenizer-12Hz...")
        print("This converts audio waveforms into discrete audio codes (16 codebook channels)")
        print("These codes are the target labels for fine-tuning.\n")
        output_path = prepare_training_data(
            input_jsonl=train_jsonl,
            output_jsonl=output_jsonl,
            tokenizer_model_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
            batch_size=getattr(args, 'tokenizer_batch_size', 32),
        )

    print(f"\n✓ Audio code extraction complete!")
    print(f"  Output: {output_path}")

    return output_path


def step_3_train(args, train_jsonl: str = None):
    """Step 3: Distributed training with Qwen3-TTS SFT + Ray Train"""
    print_header("STEP 3: Qwen3-TTS Supervised Fine-Tuning (SFT)")

    from train_qwen_tts import run_qwen_tts_training, QwenTTSConfig

    # Use the JSONL with audio codes for proper training
    train_jsonl = train_jsonl or str(PROCESSED_DIR / "train_with_codes.jsonl")

    # Check if audio codes have been prepared
    if not Path(train_jsonl).exists():
        print(f"WARNING: {train_jsonl} not found!")
        print("Running data preparation to extract audio codes...")
        train_jsonl = step_2_prepare_data(args)

    config = QwenTTSConfig(
        model_path=args.model_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        speaker_name=getattr(args, 'speaker_name', 'custom_speaker'),
    )

    print(f"\nTraining configuration:")
    print(f"  - Model: {config.model_path} (1.7B parameters)")
    print(f"  - Workers: {config.num_workers} GPUs")
    print(f"  - Batch size per GPU: {config.batch_size}")
    print(f"  - Effective batch: {config.batch_size * config.num_workers * config.gradient_accumulation_steps}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Speaker: {config.speaker_name}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"\nEstimated time: ~{config.num_epochs * 20}-{config.num_epochs * 30} minutes")

    checkpoint_path = run_qwen_tts_training(
        train_jsonl=train_jsonl,
        config=config
    )

    print(f"\n✓ Training complete!")
    print(f"  Checkpoint: {checkpoint_path}")

    return checkpoint_path


def step_4_inference(args, checkpoint_path: str = None):
    """Step 4: Inference comparison (base vs fine-tuned)"""
    print_header("STEP 4: Inference Comparison")

    from inference import run_comparison
    import json
    from pathlib import Path

    config = InferenceConfig()

    # Find reference audio
    ref_audio = args.ref_audio
    ref_text = args.ref_text or ""

    if not ref_audio:
        # Try to find from processed data
        train_jsonl = PROCESSED_DIR / "train.jsonl"
        if train_jsonl.exists():
            with open(train_jsonl) as f:
                first_entry = json.loads(f.readline())
                ref_audio = first_entry.get("ref_audio")
                ref_text = first_entry.get("text", "")

    if not ref_audio:
        print("ERROR: No reference audio found!")
        print("Please provide --ref-audio or run data processing first")
        return None

    # Use checkpoint from args if provided, otherwise use the one from training
    # Also check for final model at well-known location
    final_checkpoint = args.checkpoint or checkpoint_path

    if not final_checkpoint:
        # Look for final model at well-known location
        final_model_path = Path("/mnt/cluster_storage/qwen3-tts-training/final_model/model.pt")
        if final_model_path.exists():
            final_checkpoint = str(final_model_path)
            print(f"Found final model at: {final_checkpoint}")

    print(f"Reference audio: {ref_audio}")
    print(f"Checkpoint: {final_checkpoint or 'None (base model only)'}")

    if final_checkpoint:
        print(f"\nWill generate TWO outputs:")
        print(f"  1. Base Qwen3-TTS CustomVoice model (pre-defined speaker)")
        print(f"  2. Fine-tuned Qwen3-TTS model (your custom voice)")
    else:
        print(f"\nWill generate ONE output:")
        print(f"  1. Base Qwen3-TTS model (text-to-speech)")
        print(f"  (Provide --checkpoint to also run fine-tuned model)")

    results = run_comparison(
        test_texts=config.test_texts,
        ref_audio=ref_audio,
        ref_text=ref_text,
        checkpoint_path=final_checkpoint,
        output_dir=str(config.output_dir)
    )

    print(f"\n✓ Inference complete!")
    print(f"  Results: {config.output_dir}")

    return results


def run_tests():
    """Run all module tests"""
    print_header("Running Module Tests")

    print("\n1. Testing data_processing...")
    from data_processing import test_data_processing
    test_data_processing()

    print("\n2. Testing prepare_data...")
    from prepare_data import test_data_preparation
    test_data_preparation()

    print("\n3. Testing train_qwen_tts...")
    from train_qwen_tts import test_training_components
    test_training_components()

    print("\n4. Testing inference...")
    from inference import test_inference_functions
    test_inference_functions()

    print_header("All Tests Passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Cloning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python scripts/run_pipeline.py --all

  # Run data processing only
  python scripts/run_pipeline.py --process

  # Run training with custom settings
  python scripts/run_pipeline.py --train --num-workers 4 --epochs 5

  # Run inference comparison
  python scripts/run_pipeline.py --infer --checkpoint /path/to/checkpoint
        """
    )

    # Pipeline stages
    parser.add_argument("--all", action="store_true",
                        help="Run complete pipeline")
    parser.add_argument("--process", action="store_true",
                        help="Run data processing only")
    parser.add_argument("--prepare", action="store_true",
                        help="Run data preparation only")
    parser.add_argument("--train", action="store_true",
                        help="Run training only")
    parser.add_argument("--infer", action="store_true",
                        help="Run inference only")
    parser.add_argument("--test", action="store_true",
                        help="Run tests only")

    # Data processing options
    parser.add_argument("--input-dir", type=str,
                        help="Input directory with audio files")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory for processed data")
    parser.add_argument("--skip-tokenization", action="store_true",
                        help="Skip audio tokenization step")

    # Training options
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base model path")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of GPU workers")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per worker")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num-epochs", "-e", type=int, default=3,
                        help="Number of training epochs")

    # Inference options
    parser.add_argument("--checkpoint", type=str,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--ref-audio", type=str,
                        help="Reference audio for zero-shot")
    parser.add_argument("--ref-text", type=str,
                        help="Transcript of reference audio")

    # General options
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for single-GPU operations")

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Run appropriate stages
    if args.test:
        run_tests()
        return

    start_time = datetime.now()
    print_header(f"Qwen3-TTS Voice Cloning Pipeline")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_jsonl = None
    checkpoint_path = None

    try:
        if args.all or args.process:
            train_jsonl = step_1_process_data(args)

        if args.all or args.prepare:
            train_jsonl = step_2_prepare_data(args, train_jsonl)

        if args.all or args.train:
            checkpoint_path = step_3_train(args, train_jsonl)

        if args.all or args.infer:
            step_4_inference(args, checkpoint_path)

        # If no stage selected, show help
        if not any([args.all, args.process, args.prepare, args.train, args.infer]):
            parser.print_help()
            return

        end_time = datetime.now()
        duration = end_time - start_time

        print_header("Pipeline Complete!")
        print(f"Duration: {duration}")
        print(f"Outputs: {OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
