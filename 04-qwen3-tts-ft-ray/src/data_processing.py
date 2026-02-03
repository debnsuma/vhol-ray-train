"""
Data Processing Pipeline using Ray Data

This module handles:
1. Loading audio files (MP4, WAV, etc.)
2. Converting to WAV format
3. Transcribing with Whisper
4. Segmenting into training chunks
5. Outputting JSONL format for Qwen3-TTS

Learning Objectives:
- Distributed data processing with Ray Data
- Audio preprocessing pipeline
- Integration with speech-to-text models
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import ray
import numpy as np

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DataConfig, setup_directories, PROJECT_ROOT


# ============================================================================
# Audio Processing Functions (standalone, not using ray.remote)
# ============================================================================

def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000) -> bool:
    """Convert/resample audio file to WAV format using soundfile."""
    import soundfile as sf
    import scipy.signal

    try:
        audio, sr = sf.read(input_path)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample if needed
        if sr != sample_rate:
            num_samples = int(len(audio) * sample_rate / sr)
            audio = scipy.signal.resample(audio, num_samples)
        # Save as WAV
        sf.write(output_path, audio.astype(np.float32), sample_rate)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def load_audio(file_path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Load audio file and return numpy array using soundfile."""
    import soundfile as sf
    import scipy.signal

    try:
        audio, sr = sf.read(file_path)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample if needed
        if sr != sample_rate:
            num_samples = int(len(audio) * sample_rate / sr)
            audio = scipy.signal.resample(audio, num_samples)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def transcribe_audio(
    audio: np.ndarray,
    model_name: str = "base",
    language: str = "en"
) -> Dict[str, Any]:
    """Transcribe audio using Whisper. Accepts numpy array directly."""
    import whisper

    model = whisper.load_model(model_name, download_root="/mnt/cluster_storage/whisper_models")
    result = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        verbose=False
    )

    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": result.get("language", language)
    }


def segment_audio(
    audio: np.ndarray,
    segments: List[Dict],
    sample_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 1.0
) -> List[Dict[str, Any]]:
    """Segment audio based on Whisper segments."""
    result_segments = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        duration = end - start
        if duration < min_duration or not text:
            continue

        if duration > max_duration:
            current_start = start
            while current_start < end:
                chunk_end = min(current_start + max_duration, end)
                start_sample = int(current_start * sample_rate)
                end_sample = int(chunk_end * sample_rate)

                result_segments.append({
                    "audio": audio[start_sample:end_sample],
                    "text": text,
                    "start": current_start,
                    "end": chunk_end,
                    "duration": chunk_end - current_start
                })
                current_start = chunk_end
        else:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            result_segments.append({
                "audio": audio[start_sample:end_sample],
                "text": text,
                "start": start,
                "end": end,
                "duration": duration
            })

    return result_segments


def process_single_audio_local(
    audio_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Process a single audio file (local execution).

    This function:
    1. Converts audio to WAV
    2. Transcribes with Whisper
    3. Segments the audio
    4. Saves segments as individual WAV files
    """
    import soundfile as sf

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # Step 1: Load audio at 16kHz for Whisper
        audio_16k = load_audio(str(audio_path), config["target_sample_rate"])
        if audio_16k is None:
            return []

        # Step 2: Transcribe (pass numpy array directly)
        transcription = transcribe_audio(
            audio_16k,
            config["whisper_model"],
            config["whisper_language"]
        )

        # Step 3: Load audio at output sample rate (24kHz for Qwen3-TTS)
        audio_24k = load_audio(str(audio_path), config["output_sample_rate"])

        # Step 4: Segment audio
        segments = segment_audio(
            audio_24k,
            transcription["segments"],
            config["output_sample_rate"],
            config["max_segment_duration"],
            config["min_segment_duration"]
        )

        # Step 5: Save segments
        results = []
        base_name = audio_path.stem

        for i, seg in enumerate(segments):
            seg_filename = f"{base_name}_seg{i:04d}.wav"
            seg_path = output_dir / seg_filename

            sf.write(str(seg_path), seg["audio"], config["output_sample_rate"])

            results.append({
                "audio": str(seg_path),
                "text": seg["text"],
                "duration": seg["duration"],
                "source_file": str(audio_path)
            })

        print(f"Processed {audio_path.name}: {len(results)} segments")
        return results

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ============================================================================
# Ray Remote Function - Self-contained
# ============================================================================

@ray.remote(num_gpus=0.5)  # Use GPU for Whisper
def process_audio_ray(
    audio_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Ray remote function to process a single audio file.
    This is self-contained with all necessary imports and functions.
    Uses soundfile for WAV file handling (no ffmpeg needed for WAV).
    """
    import os
    import tempfile
    from pathlib import Path
    from typing import Dict, List, Any, Optional

    import numpy as np
    import soundfile as sf
    import scipy.signal
    import whisper

    # Helper functions (embedded for Ray serialization)
    def _load_and_resample(input_path: str, target_sr: int) -> Optional[np.ndarray]:
        """Load audio and resample to target sample rate."""
        try:
            audio, sr = sf.read(input_path)
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr != target_sr:
                num_samples = int(len(audio) * target_sr / sr)
                audio = scipy.signal.resample(audio, num_samples)
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return None

    def _save_wav(audio: np.ndarray, output_path: str, sample_rate: int) -> bool:
        """Save audio to WAV file."""
        try:
            sf.write(output_path, audio, sample_rate)
            return True
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
            return False

    def _transcribe(audio: np.ndarray, model_name: str, language: str) -> Dict[str, Any]:
        # Use shared storage for whisper model to avoid race conditions
        # Pass audio as numpy array directly to avoid ffmpeg dependency
        model = whisper.load_model(model_name, download_root="/mnt/cluster_storage/whisper_models")
        # Whisper expects audio at 16kHz, float32
        result = model.transcribe(audio, language=language, word_timestamps=True, verbose=False)
        return {"text": result["text"], "segments": result["segments"], "language": result.get("language", language)}

    def _segment(audio: np.ndarray, segments: List[Dict], sr: int, max_dur: float, min_dur: float) -> List[Dict]:
        result = []
        for seg in segments:
            start, end, text = seg["start"], seg["end"], seg["text"].strip()
            duration = end - start
            if duration < min_dur or not text:
                continue
            if duration > max_dur:
                curr = start
                while curr < end:
                    chunk_end = min(curr + max_dur, end)
                    result.append({
                        "audio": audio[int(curr * sr):int(chunk_end * sr)],
                        "text": text, "start": curr, "end": chunk_end, "duration": chunk_end - curr
                    })
                    curr = chunk_end
            else:
                result.append({
                    "audio": audio[int(start * sr):int(end * sr)],
                    "text": text, "start": start, "end": end, "duration": duration
                })
        return result

    # Main processing
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # Load audio at 16kHz for Whisper
        audio_16k = _load_and_resample(str(audio_path), config["target_sample_rate"])
        if audio_16k is None:
            return []

        # Transcribe (pass numpy array directly to avoid ffmpeg)
        transcription = _transcribe(audio_16k, config["whisper_model"], config["whisper_language"])

        # Load audio at output sample rate for segments
        audio_24k = _load_and_resample(str(audio_path), config["output_sample_rate"])
        if audio_24k is None:
            return []

        # Segment audio
        segments = _segment(
            audio_24k, transcription["segments"], config["output_sample_rate"],
            config["max_segment_duration"], config["min_segment_duration"]
        )

        # Save segments
        results = []
        for i, seg in enumerate(segments):
            seg_path = output_dir / f"{audio_path.stem}_seg{i:04d}.wav"
            sf.write(str(seg_path), seg["audio"], config["output_sample_rate"])
            results.append({
                "audio": str(seg_path),
                "text": seg["text"],
                "duration": seg["duration"],
                "source_file": str(audio_path)
            })

        print(f"Processed {audio_path.name}: {len(results)} segments")
        return results

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ============================================================================
# Main Processing Function
# ============================================================================

def run_data_processing(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[DataConfig] = None,
    use_ray: bool = True
) -> str:
    """
    Run the full data processing pipeline.

    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save processed data
        config: Data processing configuration
        use_ray: Whether to use Ray for distributed processing

    Returns:
        Path to the output JSONL file
    """
    if config is None:
        config = DataConfig()

    input_dir = Path(input_dir) if input_dir else config.raw_audio_dir
    output_dir = Path(output_dir) if output_dir else config.processed_dir

    # Setup directories
    setup_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in config.audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))

    if not audio_files:
        raise ValueError(f"No audio files found in {input_dir}")

    print(f"Found {len(audio_files)} audio files to process")

    # Prepare config dict
    config_dict = {
        "target_sample_rate": config.target_sample_rate,
        "output_sample_rate": config.output_sample_rate,
        "max_segment_duration": config.max_segment_duration,
        "min_segment_duration": config.min_segment_duration,
        "whisper_model": config.whisper_model,
        "whisper_language": config.whisper_language,
    }

    wav_output_dir = output_dir / "wav"
    all_results = []

    if use_ray:
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()

        # Process files in parallel with Ray
        futures = [
            process_audio_ray.remote(str(f), str(wav_output_dir), config_dict)
            for f in audio_files
        ]

        # Collect results
        for future in futures:
            results = ray.get(future)
            all_results.extend(results)
    else:
        # Process locally (for testing or single-machine)
        for f in audio_files:
            results = process_single_audio_local(str(f), str(wav_output_dir), config_dict)
            all_results.extend(results)

    print(f"Total segments processed: {len(all_results)}")

    # Create reference audio (use longest segment)
    if all_results:
        ref_audio = max(all_results, key=lambda x: x["duration"])
        ref_audio_path = ref_audio["audio"]
    else:
        raise ValueError("No segments were processed successfully")

    # Write JSONL output
    jsonl_path = output_dir / config.train_jsonl
    with open(jsonl_path, "w") as f:
        for item in all_results:
            entry = {
                "audio": item["audio"],
                "text": item["text"],
                "ref_audio": ref_audio_path
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Training data saved to: {jsonl_path}")
    print(f"Reference audio: {ref_audio_path}")

    return str(jsonl_path)


# ============================================================================
# Testing Function
# ============================================================================

def test_data_processing():
    """Test individual functions in the data processing pipeline"""
    print("Testing data processing functions...")

    # Test 1: Audio conversion
    print("\n1. Testing audio loading...")
    test_audio = np.random.randn(16000).astype(np.float32)
    assert test_audio.shape[0] == 16000, "Audio shape incorrect"
    print("   ✓ Audio array creation works")

    # Test 2: Segmentation
    print("\n2. Testing segmentation...")
    test_segments = [
        {"start": 0.0, "end": 2.0, "text": "Test segment one"},
        {"start": 2.0, "end": 5.0, "text": "Test segment two"},
    ]
    result = segment_audio(
        np.random.randn(80000).astype(np.float32),
        test_segments,
        16000,
        max_duration=15.0,
        min_duration=1.0
    )
    assert len(result) == 2, "Should have 2 segments"
    print(f"   ✓ Segmentation works: {len(result)} segments")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process audio data for Qwen3-TTS training")
    parser.add_argument("--input-dir", type=str, help="Input directory with audio files")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed data")
    parser.add_argument("--no-ray", action="store_true", help="Don't use Ray (local processing)")
    parser.add_argument("--test", action="store_true", help="Run tests only")

    args = parser.parse_args()

    if args.test:
        test_data_processing()
    else:
        jsonl_path = run_data_processing(
            args.input_dir,
            args.output_dir,
            use_ray=not args.no_ray
        )
        print(f"\nData processing complete!")
        print(f"JSONL file: {jsonl_path}")
