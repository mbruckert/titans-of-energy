# silence_removal.py
import sys
import librosa
import numpy as np
from typing import Tuple, Optional
import soundfile as sf

def load_audio(file_path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """Load audio file and optionally resample."""
    signal, original_sr = librosa.load(file_path, sr=sr)
    return signal, original_sr

def remove_silence_v3(
    signal: np.ndarray,
    top_db: float = 40.0,
    frame_length: int = 4096,
    hop_length: int = 1024,
    fade_length_ms: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Remove silence from audio signal with natural transitions and track timing."""
    # Calculate fade length in samples
    fade_samples = int(fade_length_ms * len(signal) / (1000 * len(signal)))

    # Find silent regions
    intervals = librosa.effects.split(
        signal,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Initialize output array
    processed_signal = np.zeros_like(signal)
    current_pos = 0

    # Process each segment
    for i, (start, end) in enumerate(intervals):
        segment = signal[start:end]

        if i == 0:
            processed_signal[:len(segment)] = segment
            current_pos = len(segment)
            continue

        prev_segment = signal[current_pos - fade_samples:current_pos]
        next_segment = segment[:fade_samples]

        if len(prev_segment) > 0 and len(next_segment) > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            crossfaded = prev_segment * fade_out + next_segment * fade_in

            processed_signal[current_pos-fade_samples:current_pos] = crossfaded

        segment_start = current_pos
        segment_end = current_pos + len(segment) - fade_samples
        processed_signal[segment_start:segment_end] = segment[fade_samples:]

        current_pos = segment_end

    # Trim trailing zeros
    processed_signal = processed_signal[:current_pos]

    # Calculate actual duration
    total_duration_seconds = current_pos / len(signal)

    return processed_signal, intervals, total_duration_seconds

def process_audio(
    input_path: str,
    output_path: str,
    sr: Optional[int] = None,
    top_db: float = 40.0,
    fade_length_ms: int = 50
):
    """Main function to process audio file."""
    signal, original_sr = load_audio(input_path, sr=sr)

    # Remove silence and get timing info
    signal_no_silence, _, duration_ratio = remove_silence_v3(
        signal,
        top_db=top_db,
        fade_length_ms=fade_length_ms
    )

    sf.write(output_path, signal_no_silence, original_sr)

def main():
    if len(sys.argv) < 3:
        print("Usage: python silence_removal.py <input_file> <output_file> [options]")
        print("Options:")
        print("  --top_db <float>    Silence threshold in dB (default: 40.0)")
        print("  --fade_ms <int>     Fade length in milliseconds (default: 50)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse optional arguments
    top_db = 40.0
    fade_ms = 50

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--top_db" and i + 1 < len(sys.argv):
            top_db = float(sys.argv[i + 1])
        elif sys.argv[i] == "--fade_ms" and i + 1 < len(sys.argv):
            fade_ms = int(sys.argv[i + 1])

    process_audio(input_file, output_file, top_db=top_db, fade_length_ms=fade_ms)

if __name__ == "__main__":
    main()
