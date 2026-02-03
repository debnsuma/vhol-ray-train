#!/usr/bin/env python3
"""
Zero-Shot Voice Cloning with Qwen3-TTS Base Model

This script performs zero-shot voice cloning - it takes a reference audio sample
of any voice and generates new speech in that voice without any fine-tuning.

Usage:
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

Modes:
    - X-Vector Only Mode (default): Uses only speaker embedding from reference audio.
      Simpler, doesn't require reference text transcript.

    - ICL Mode (--ref-text provided): Uses In-Context Learning with reference audio
      AND its transcript. Generally produces better voice matching.

Examples:
    # Clone a voice with just a sample
    python scripts/zero_shot_clone.py \\
        --ref-audio samples/my_voice.wav \\
        --text "Hello, this is my cloned voice speaking." \\
        --output output/cloned_speech.wav

    # Better quality with reference transcript
    python scripts/zero_shot_clone.py \\
        --ref-audio samples/my_voice.wav \\
        --ref-text "This is what I said in the reference audio." \\
        --text "Now I can say anything in this voice." \\
        --output output/cloned_speech.wav
"""

import argparse
import os
import sys
from pathlib import Path


def clone_voice_local(
    ref_audio: str,
    text: str,
    output_path: str,
    ref_text: str = None,
    language: str = "english",
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> dict:
    """
    Perform zero-shot voice cloning locally (requires GPU).

    Args:
        ref_audio: Path to reference audio file (the voice to clone)
        text: Text to synthesize
        output_path: Path to save generated audio
        ref_text: Optional transcript of reference audio (enables ICL mode)
        language: Language for synthesis
        model_path: HuggingFace model path

    Returns:
        dict with success status and output path
    """
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    result = {"success": False, "output_path": output_path, "error": None}

    try:
        print(f"Loading model: {model_path}")
        wrapper = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        # Determine mode based on whether ref_text is provided
        x_vector_only = ref_text is None or ref_text.strip() == ""
        mode = "x-vector only" if x_vector_only else "ICL (in-context learning)"

        print(f"Reference audio: {ref_audio}")
        print(f"Mode: {mode}")
        print(f"Language: {language}")
        print(f"Text to synthesize: '{text[:80]}{'...' if len(text) > 80 else ''}'")

        with torch.no_grad():
            wavs, sr = wrapper.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
            )

        # Process and save output
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)

        result["success"] = True
        result["sample_rate"] = sr
        result["mode"] = mode
        print(f"\nSaved: {output_path}")
        print(f"Sample rate: {sr} Hz")

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
        traceback.print_exc()

    return result


def clone_voice_ray(
    ref_audio: str,
    text: str,
    output_path: str,
    ref_text: str = None,
    language: str = "english",
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> dict:
    """
    Perform zero-shot voice cloning on a Ray GPU worker.

    The reference audio is loaded on the head node and passed as numpy array
    to the worker, so local files work even if not on shared storage.
    """
    import ray
    import numpy as np
    import librosa

    # Load audio on head node (where the file exists)
    print(f"Loading reference audio: {ref_audio}")
    ref_audio_data, ref_sr = librosa.load(ref_audio, sr=None, mono=True)
    print(f"  Sample rate: {ref_sr} Hz, Duration: {len(ref_audio_data)/ref_sr:.2f}s")

    @ray.remote(num_gpus=1)
    def _clone_on_gpu(ref_audio_data, ref_sr, text, output_path, ref_text, language, model_path):
        import torch
        import soundfile as sf
        from pathlib import Path
        from qwen_tts import Qwen3TTSModel

        result = {"success": False, "output_path": output_path, "error": None}

        try:
            print(f"Loading model: {model_path}")
            wrapper = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )

            x_vector_only = ref_text is None or ref_text.strip() == ""
            mode = "x-vector only" if x_vector_only else "ICL (in-context learning)"

            print(f"Mode: {mode}")
            print(f"Language: {language}")
            print(f"Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")

            # Pass audio as tuple (numpy_array, sample_rate)
            ref_audio_input = (ref_audio_data, ref_sr)

            with torch.no_grad():
                wavs, sr = wrapper.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio_input,
                    ref_text=ref_text if not x_vector_only else None,
                    x_vector_only_mode=x_vector_only,
                )

            audio = wavs[0] if isinstance(wavs, list) else wavs
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()

            result["success"] = True
            result["sample_rate"] = sr
            result["mode"] = mode
            result["audio_data"] = audio  # Return audio data to head node
            print(f"Generated audio successfully")

        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"Error: {e}")
            traceback.print_exc()

        return result

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()

    print("Submitting to Ray GPU worker...")
    future = _clone_on_gpu.remote(
        ref_audio_data, ref_sr, text, output_path, ref_text, language, model_path
    )
    result = ray.get(future)

    # Save output on head node (where the user ran the command)
    if result["success"] and "audio_data" in result:
        import soundfile as sf
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, result["audio_data"], result["sample_rate"])
        result["output_path"] = output_path
        del result["audio_data"]  # Don't keep large array in result
        print(f"Saved: {output_path}")

    return result


def resolve_path(path: str, must_exist: bool = False) -> str:
    """
    Convert relative paths to absolute paths.

    This ensures paths work correctly when running on Ray workers
    which may have different working directories.
    """
    if path is None:
        return None

    # Already absolute
    if os.path.isabs(path):
        return path

    # Convert to absolute path based on current working directory
    abs_path = os.path.abspath(path)

    if must_exist and not os.path.exists(abs_path):
        return abs_path  # Return anyway, let caller handle the error

    return abs_path


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Voice Cloning with Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic voice cloning
  python scripts/zero_shot_clone.py \\
      --ref-audio voice_sample.wav \\
      --text "Hello world" \\
      --output output.wav

  # With reference transcript (better quality)
  python scripts/zero_shot_clone.py \\
      --ref-audio voice_sample.wav \\
      --ref-text "What was said in the sample" \\
      --text "New text to speak" \\
      --output output.wav
        """
    )

    parser.add_argument(
        "--ref-audio", "-r",
        type=str,
        required=True,
        help="Path to reference audio file (the voice to clone)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to synthesize in the cloned voice"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for generated audio (default: ./output.wav)"
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of reference audio (enables ICL mode for better quality)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="english",
        choices=["auto", "english", "chinese", "japanese", "korean",
                 "german", "french", "spanish", "italian", "portuguese", "russian"],
        help="Language for synthesis (default: english)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on Ray cluster (requires local GPU)"
    )

    args = parser.parse_args()

    # Convert relative paths to absolute paths
    # This is critical for Ray workers which run in different directories
    ref_audio = resolve_path(args.ref_audio)

    # Default output path is ./output.wav in current working directory
    if args.output is None:
        output_path = os.path.join(os.getcwd(), "output.wav")
    else:
        output_path = resolve_path(args.output)

    # Validate reference audio exists
    if not os.path.exists(ref_audio):
        print(f"Error: Reference audio not found: {ref_audio}")
        sys.exit(1)

    # Ensure output directory exists locally (will be created on worker too)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Zero-Shot Voice Cloning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Reference audio: {ref_audio}")
    if args.ref_text:
        print(f"Reference text: '{args.ref_text[:50]}...'")
        print(f"Mode: ICL (In-Context Learning)")
    else:
        print(f"Mode: X-Vector Only (speaker embedding)")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Run voice cloning
    if args.local:
        result = clone_voice_local(
            ref_audio=ref_audio,
            text=args.text,
            output_path=output_path,
            ref_text=args.ref_text,
            language=args.language,
            model_path=args.model,
        )
    else:
        result = clone_voice_ray(
            ref_audio=ref_audio,
            text=args.text,
            output_path=output_path,
            ref_text=args.ref_text,
            language=args.language,
            model_path=args.model,
        )

    print()
    print("=" * 60)
    if result["success"]:
        print("SUCCESS!")
        print(f"Output: {result['output_path']}")
        print(f"Mode: {result.get('mode', 'unknown')}")
    else:
        print("FAILED!")
        print(f"Error: {result['error']}")
    print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
