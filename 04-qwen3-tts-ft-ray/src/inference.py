"""
Inference and Comparison Module

This module provides:
1. Zero-shot voice cloning with Qwen3-TTS Base model (clones your voice from reference audio)
2. Voice cloning with fine-tuned Qwen3-TTS model (improved voice matching after training)
3. Side-by-side comparison of base vs fine-tuned model outputs

All inference runs on GPU worker nodes via Ray.

IMPORTANT: Both models use generate_voice_clone() to clone your voice from reference audio.
The fine-tuned model should produce better voice matching than the base model.
"""

import os
import sys
import json
import ray
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import InferenceConfig, ModelConfig, PROCESSED_DIR, OUTPUT_DIR


# ============================================================================
# Ray Remote Inference Functions (run on GPU workers)
# ============================================================================

@ray.remote(num_gpus=1)
def run_voice_clone_inference(
    ref_audio_data: np.ndarray,
    ref_sr: int,
    text: str,
    language: str = "english",
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> Dict[str, Any]:
    """
    Run zero-shot voice cloning on a GPU worker using the Base model.

    This uses generate_voice_clone() to clone the voice from reference audio.
    The reference audio is passed as numpy array (loaded on head node).

    Args:
        ref_audio_data: Reference audio as numpy array
        ref_sr: Sample rate of reference audio
        text: Text to synthesize
        language: Language for synthesis (full name like "english")
        model_path: HuggingFace model path for Base model
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    result = {
        "success": False,
        "error": None,
        "audio_data": None,
        "sample_rate": None,
    }

    try:
        print(f"Loading model: {model_path}")
        wrapper = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        print(f"Generating voice clone: '{text[:50]}...'")
        print(f"Language: {language}")

        # Pass audio as tuple (numpy_array, sample_rate)
        ref_audio_input = (ref_audio_data, ref_sr)

        with torch.no_grad():
            wavs, sr = wrapper.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_input,
                x_vector_only_mode=True,  # Use speaker embedding only
            )

        audio = wavs[0] if isinstance(wavs, list) else wavs
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()

        result["success"] = True
        result["audio_data"] = audio
        result["sample_rate"] = sr
        result["model"] = model_path
        print(f"Generated audio successfully (voice clone)")

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"Inference error: {e}")
        traceback.print_exc()

    return result


@ray.remote(num_gpus=1)
def run_finetuned_voice_clone_inference(
    ref_audio_data: np.ndarray,
    ref_sr: int,
    text: str,
    checkpoint_path: str,
    language: str = "english",
) -> Dict[str, Any]:
    """
    Run voice cloning with fine-tuned Qwen3-TTS model.

    This loads the fine-tuned checkpoint and uses generate_voice_clone()
    to generate speech with your custom voice.

    Args:
        ref_audio_data: Reference audio as numpy array
        ref_sr: Sample rate of reference audio
        text: Text to synthesize
        checkpoint_path: Path to fine-tuned model checkpoint
        language: Language for synthesis
    """
    import torch
    import os
    from qwen_tts import Qwen3TTSModel

    result = {
        "success": False,
        "error": None,
        "audio_data": None,
        "sample_rate": None,
        "checkpoint_path": checkpoint_path,
    }

    try:
        device = torch.device("cuda:0")

        # Find checkpoint file
        checkpoint_file = None
        if os.path.isdir(checkpoint_path):
            potential_files = ["checkpoint.pt", "model.pt", "checkpoint.pth"]
            for f in potential_files:
                fp = os.path.join(checkpoint_path, f)
                if os.path.exists(fp):
                    checkpoint_file = fp
                    break
        else:
            checkpoint_file = checkpoint_path

        if not checkpoint_file or not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        # Load base model
        base_model_path = checkpoint.get("config", {}).get("model_path", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        print(f"Loading base model: {base_model_path}")

        wrapper = Qwen3TTSModel.from_pretrained(
            base_model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        model = wrapper.model

        # Load fine-tuned weights
        if "model_state_dict" in checkpoint:
            print("Loading fine-tuned weights...")
            state_dict = checkpoint["model_state_dict"]

            # Handle DTensor conversion if needed
            converted_state_dict = {}
            for key, value in state_dict.items():
                if hasattr(value, '_local_tensor'):
                    converted_state_dict[key] = value._local_tensor.clone().detach().to(device)
                elif torch.is_tensor(value):
                    converted_state_dict[key] = value.to(device)
                else:
                    converted_state_dict[key] = value

            # Load with strict=False to handle any architecture differences
            missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 0)
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            print(f"Loaded fine-tuned model from epoch {epoch}, loss: {loss_str}")
            result["model_epoch"] = epoch
            result["model_loss"] = loss

        model.eval()

        # Generate audio using voice cloning
        print(f"Generating voice clone: '{text[:50]}...'")
        print(f"Language: {language}")

        # Pass audio as tuple (numpy_array, sample_rate)
        ref_audio_input = (ref_audio_data, ref_sr)

        with torch.no_grad():
            wavs, sr = wrapper.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_input,
                x_vector_only_mode=True,  # Use speaker embedding only
            )

        audio = wavs[0] if isinstance(wavs, list) else wavs
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()

        result["success"] = True
        result["audio_data"] = audio
        result["sample_rate"] = sr
        result["speaker"] = checkpoint.get("config", {}).get("speaker_name", "custom_speaker")
        print(f"Generated audio successfully (fine-tuned voice clone)")

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"Fine-tuned inference error: {e}")
        traceback.print_exc()

    return result


# ============================================================================
# Comparison Pipeline
# ============================================================================

def run_comparison(
    test_texts: List[str],
    ref_audio: str,
    ref_text: str,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[InferenceConfig] = None,
    model_config: Optional[ModelConfig] = None
) -> Dict[str, Any]:
    """
    Run inference comparison on GPU workers via Ray.

    Generates TWO outputs for comparison:
    1. Base model output: Zero-shot voice cloning with Qwen3-TTS Base model
    2. Fine-tuned model output: Voice cloning with your fine-tuned model

    Both use generate_voice_clone() to clone your voice from reference audio.
    The fine-tuned model should produce better voice matching.

    Args:
        test_texts: List of texts to synthesize
        ref_audio: Reference audio path for voice cloning
        ref_text: Reference audio transcript (for metadata)
        checkpoint_path: Path to fine-tuned Qwen3-TTS checkpoint
        output_dir: Directory to save outputs
        config: Inference configuration
        model_config: Model configuration
    """
    import soundfile as sf

    if config is None:
        config = InferenceConfig()
    if model_config is None:
        model_config = ModelConfig()

    output_dir = Path(output_dir) if output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()

    # Load reference audio on head node (so it's accessible regardless of path)
    print(f"Loading reference audio: {ref_audio}")
    ref_audio_data, ref_sr = librosa.load(ref_audio, sr=None, mono=True)
    print(f"  Sample rate: {ref_sr} Hz, Duration: {len(ref_audio_data)/ref_sr:.2f}s")

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_texts": test_texts,
        "ref_audio": ref_audio,
        "checkpoint_path": checkpoint_path,
        "outputs": []
    }

    # Use Base model for voice cloning (not CustomVoice which only has predefined speakers)
    base_model_path = model_config.base_model

    print(f"\n{'='*60}")
    print("Running Voice Cloning Comparison")
    print(f"{'='*60}")
    print(f"Base Model (zero-shot): {base_model_path}")
    print(f"Fine-tuned Checkpoint: {checkpoint_path or 'None'}")
    print(f"Reference audio: {ref_audio}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Process each test text
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Processing text {i+1}/{len(test_texts)}:")
        print(f"  '{text[:80]}...'")
        print(f"{'='*60}")

        text_results = {
            "text": text,
            "outputs": {}
        }

        # ================================================================
        # 1. Generate with BASE model (zero-shot voice cloning)
        # ================================================================
        base_output_path = str(output_dir / f"text{i+1}_base_model.wav")

        print("\n[1/2] Generating with BASE model (zero-shot voice cloning)...")
        print(f"      Model: {base_model_path}")

        base_future = run_voice_clone_inference.remote(
            ref_audio_data=ref_audio_data,
            ref_sr=ref_sr,
            text=text,
            language="english",
            model_path=base_model_path,
        )

        base_result = ray.get(base_future)

        if base_result["success"]:
            # Save audio on head node
            sf.write(base_output_path, base_result["audio_data"], base_result["sample_rate"])
            text_results["outputs"]["base_model"] = {
                "path": base_output_path,
                "model": base_model_path,
                "method": "zero-shot voice clone",
            }
            print(f"      ✓ Base model generated: {base_output_path}")
        else:
            text_results["outputs"]["base_model"] = {
                "path": None,
                "error": base_result["error"]
            }
            print(f"      ✗ Base model failed: {base_result['error']}")

        # ================================================================
        # 2. Generate with FINE-TUNED model (if checkpoint provided)
        # ================================================================
        if checkpoint_path:
            finetuned_output_path = str(output_dir / f"text{i+1}_finetuned_model.wav")

            print(f"\n[2/2] Generating with FINE-TUNED model (voice cloning)...")
            print(f"      Checkpoint: {checkpoint_path}")

            finetuned_future = run_finetuned_voice_clone_inference.remote(
                ref_audio_data=ref_audio_data,
                ref_sr=ref_sr,
                text=text,
                checkpoint_path=checkpoint_path,
                language="english",
            )

            finetuned_result = ray.get(finetuned_future)

            if finetuned_result["success"]:
                # Save audio on head node
                sf.write(finetuned_output_path, finetuned_result["audio_data"], finetuned_result["sample_rate"])
                text_results["outputs"]["finetuned_model"] = {
                    "path": finetuned_output_path,
                    "checkpoint": checkpoint_path,
                    "model_epoch": finetuned_result.get("model_epoch"),
                    "model_loss": finetuned_result.get("model_loss"),
                    "speaker": finetuned_result.get("speaker"),
                    "method": "fine-tuned voice clone",
                }
                print(f"      ✓ Fine-tuned model generated: {finetuned_output_path}")
                print(f"        Epoch: {finetuned_result.get('model_epoch')}, "
                      f"Loss: {finetuned_result.get('model_loss')}")
            else:
                text_results["outputs"]["finetuned_model"] = {
                    "path": None,
                    "error": finetuned_result["error"]
                }
                print(f"      ✗ Fine-tuned model failed: {finetuned_result['error']}")
        else:
            print(f"\n[2/2] Skipping fine-tuned model (no checkpoint provided)")
            text_results["outputs"]["finetuned_model"] = {
                "path": None,
                "note": "No checkpoint provided"
            }

        results["outputs"].append(text_results)

    # Save results metadata
    meta_path = output_dir / "comparison_results.json"
    with open(meta_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("VOICE CLONING COMPARISON COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput files:")

    for i, output in enumerate(results["outputs"]):
        print(f"\n  Text {i+1}:")
        if output["outputs"].get("base_model", {}).get("path"):
            print(f"    - Base model (zero-shot):  {output['outputs']['base_model']['path']}")
        if output["outputs"].get("finetuned_model", {}).get("path"):
            print(f"    - Fine-tuned model:        {output['outputs']['finetuned_model']['path']}")

    print(f"\nMetadata: {meta_path}")
    print(f"{'='*60}\n")

    return results


# ============================================================================
# Quick Inference Function
# ============================================================================

def quick_inference(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str,
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
) -> str:
    """
    Quick single-text voice cloning inference via Ray GPU worker.
    """
    import soundfile as sf

    if not ray.is_initialized():
        ray.init()

    # Load reference audio on head node
    print(f"Loading reference audio: {ref_audio}")
    ref_audio_data, ref_sr = librosa.load(ref_audio, sr=None, mono=True)

    print(f"Submitting voice clone inference to GPU worker...")
    future = run_voice_clone_inference.remote(
        ref_audio_data=ref_audio_data,
        ref_sr=ref_sr,
        text=text,
        language="english",
        model_path=model_path,
    )

    result = ray.get(future)

    if result["success"]:
        # Save audio on head node
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, result["audio_data"], result["sample_rate"])
        print(f"Generated audio saved to: {output_path}")
        return output_path
    else:
        print(f"Inference failed: {result['error']}")
        return None


# ============================================================================
# Testing Function
# ============================================================================

def test_inference_functions():
    """Test inference functions (without model loading)"""
    print("Testing inference functions...")

    # Test 1: Configuration
    print("\n1. Testing configuration...")
    config = InferenceConfig()
    assert len(config.test_texts) > 0, "Should have test texts"
    print(f"   ✓ Config has {len(config.test_texts)} test texts")

    # Test 2: Output directory creation
    print("\n2. Testing output directory...")
    test_dir = Path("/tmp/inference_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    assert test_dir.exists(), "Directory should exist"
    print(f"   ✓ Output directory works")

    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Cloning Inference")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--ref-audio", type=str, help="Reference audio path")
    parser.add_argument("--ref-text", type=str, default="", help="Reference audio transcript")
    parser.add_argument("--checkpoint", type=str, help="Fine-tuned checkpoint path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR / "inference"))
    parser.add_argument("--test", action="store_true", help="Run tests only")

    args = parser.parse_args()

    if args.test:
        test_inference_functions()
    elif args.text and args.ref_audio:
        # Quick single inference
        quick_inference(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            output_path=f"{args.output_dir}/output.wav"
        )
    else:
        # Full comparison
        config = InferenceConfig()

        # Find reference audio from processed data
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
            print("Please provide --ref-audio or run data processing first")
            sys.exit(1)

        run_comparison(
            test_texts=config.test_texts,
            ref_audio=ref_audio,
            ref_text=ref_text,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )
