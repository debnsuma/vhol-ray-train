"""
Data Preparation for Qwen3-TTS Fine-tuning

This module extracts audio codes from audio files using the Qwen3-TTS tokenizer.
The audio codes are required for fine-tuning the Qwen3-TTS model.

Steps:
1. Load audio files from train.jsonl
2. Extract audio codes using Qwen3-TTS-Tokenizer-12Hz on GPU
3. Save processed data with audio codes to output JSONL

This runs on GPU workers via Ray for efficient tokenization.
"""

import os
import sys
import json
import ray
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROCESSED_DIR, RAY_STORAGE_PATH


@ray.remote(num_gpus=1)
def extract_audio_codes_batch(
    batch_items: List[Dict[str, Any]],
    tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
) -> List[Dict[str, Any]]:
    """
    Extract audio codes from a batch of audio files using Qwen3-TTS tokenizer.

    This runs on a GPU worker for efficient processing.
    The tokenizer converts audio waveforms into discrete audio codes (16 codebook channels).

    Args:
        batch_items: List of dictionaries with 'audio' and optionally 'ref_audio' paths
        tokenizer_model_path: HuggingFace model path for tokenizer

    Returns:
        List of dictionaries with 'audio_codes' added
    """
    import torch
    import librosa
    import numpy as np

    results = []

    try:
        from qwen_tts import Qwen3TTSTokenizer

        print(f"Loading tokenizer: {tokenizer_model_path}")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_model_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )

        # Cache for reference audio codes (usually same ref_audio for all samples)
        ref_audio_cache = {}

        # Process each item in the batch
        for item in batch_items:
            try:
                audio_path = item["audio"]

                # Load audio at original sample rate
                audio, sr = librosa.load(audio_path, sr=None)

                # Encode audio to get codes
                # The tokenizer expects: audios (np.ndarray or list), sr (int)
                # Returns audio_codes with shape [time, 16] (16 codebook channels)
                with torch.no_grad():
                    encode_result = tokenizer.encode(audio, sr=sr)
                    audio_codes = encode_result["audio_codes"][0]  # [time, 16]
                    audio_codes_list = audio_codes.cpu().tolist()

                # Also process reference audio if present
                ref_audio = item.get("ref_audio")
                ref_codes_list = None
                if ref_audio:
                    if ref_audio not in ref_audio_cache:
                        ref_audio_data, ref_sr = librosa.load(ref_audio, sr=None)
                        with torch.no_grad():
                            ref_result = tokenizer.encode(ref_audio_data, sr=ref_sr)
                            ref_codes = ref_result["audio_codes"][0]
                            ref_audio_cache[ref_audio] = ref_codes.cpu().tolist()
                    ref_codes_list = ref_audio_cache[ref_audio]

                # Add audio codes to item
                result_item = item.copy()
                result_item["audio_codes"] = audio_codes_list
                if ref_codes_list:
                    result_item["ref_audio_codes"] = ref_codes_list
                results.append(result_item)

                print(f"  Processed: {Path(audio_path).name}, codes shape: {len(audio_codes_list)}x16")

            except Exception as e:
                print(f"  Error processing {item.get('audio', 'unknown')}: {e}")
                # Still include the item but without codes
                result_item = item.copy()
                result_item["audio_codes"] = None
                result_item["error"] = str(e)
                results.append(result_item)

        # Clean up GPU memory
        del tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        # Return items with error markers
        for item in batch_items:
            result_item = item.copy()
            result_item["audio_codes"] = None
            result_item["error"] = str(e)
            results.append(result_item)

    return results


def prepare_training_data(
    input_jsonl: str,
    output_jsonl: str,
    tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    batch_size: int = 32,
) -> str:
    """
    Prepare training data by extracting audio codes from all audio files.

    Args:
        input_jsonl: Path to input JSONL with audio paths and transcripts
        output_jsonl: Path to output JSONL with audio codes added
        tokenizer_model_path: HuggingFace model path for tokenizer
        batch_size: Number of items to process per GPU batch

    Returns:
        Path to output JSONL file
    """
    print(f"\n{'='*60}")
    print("Preparing Training Data with Audio Code Extraction")
    print(f"{'='*60}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")
    print(f"Tokenizer: {tokenizer_model_path}")
    print(f"Batch size: {batch_size}")

    # Load all items from input JSONL
    items = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))

    print(f"Loaded {len(items)} items from input JSONL")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Split into batches and process in parallel
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    print(f"Processing {len(batches)} batches on GPU workers...")

    # Submit all batches
    futures = [
        extract_audio_codes_batch.remote(batch, tokenizer_model_path)
        for batch in batches
    ]

    # Collect results
    all_results = []
    for i, future in enumerate(futures):
        batch_results = ray.get(future)
        all_results.extend(batch_results)
        print(f"Completed batch {i+1}/{len(batches)}")

    # Filter out items with errors
    valid_results = [r for r in all_results if r.get("audio_codes") is not None]
    error_count = len(all_results) - len(valid_results)

    print(f"\nProcessed {len(valid_results)} items successfully, {error_count} errors")

    # Write output JSONL
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in valid_results:
            f.write(json.dumps(item) + '\n')

    print(f"Saved to: {output_jsonl}")
    print(f"{'='*60}\n")

    return str(output_path)


def prepare_data_simple(input_jsonl: str, output_jsonl: str) -> str:
    """
    Simple data preparation without tokenization (for testing).
    Just copies the JSONL file.
    """
    import shutil
    shutil.copy2(input_jsonl, output_jsonl)
    print(f"Copied {input_jsonl} to {output_jsonl}")
    return output_jsonl


def test_data_preparation():
    """Test data preparation functions (without actual tokenization)"""
    print("Testing data preparation functions...")

    # Test 1: Configuration
    print("\n1. Testing configuration...")
    print(f"   ✓ Processed dir: {PROCESSED_DIR}")

    # Test 2: Simple copy
    print("\n2. Testing simple copy...")
    test_input = "/tmp/test_input.jsonl"
    test_output = "/tmp/test_output.jsonl"

    # Create test input
    with open(test_input, 'w') as f:
        f.write(json.dumps({"audio": "/test/audio.wav", "text": "test"}) + '\n')

    result = prepare_data_simple(test_input, test_output)
    assert Path(result).exists(), "Output should exist"
    print(f"   ✓ Simple copy works")

    # Cleanup
    Path(test_input).unlink(missing_ok=True)
    Path(test_output).unlink(missing_ok=True)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for Qwen3-TTS fine-tuning")
    parser.add_argument("--input", type=str, default=str(PROCESSED_DIR / "train.jsonl"),
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "train_with_codes.jsonl"),
                        help="Output JSONL file with audio codes")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        help="Tokenizer model path")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--test", action="store_true", help="Run tests only")

    args = parser.parse_args()

    if args.test:
        test_data_preparation()
    else:
        prepare_training_data(
            input_jsonl=args.input,
            output_jsonl=args.output,
            tokenizer_model_path=args.tokenizer,
            batch_size=args.batch_size,
        )
