# audio_enhance.py
import sys
import librosa
import numpy as np
from typing import Tuple, Optional
import soundfile as sf

def enhance_quality(
    signal: np.ndarray,
    sample_rate: int,
    bass_boost: bool = True,
    treble_boost: bool = True,
    compression: bool = True
) -> np.ndarray:
    """Enhance audio quality with various effects."""
    enhanced_signal = signal.copy()

    if bass_boost:
        # Bass boost around 200 Hz
        enhanced_signal = add_frequency_boost(
            enhanced_signal,
            sample_rate,
            freq_range=(150, 250),
            gain=3.0
        )

    if treble_boost:
        # Treble boost above 5000 Hz
        enhanced_signal = add_frequency_boost(
            enhanced_signal,
            sample_rate,
            freq_range=(5000, sample_rate//2),
            gain=2.0
        )

    if compression:
        enhanced_signal = dynamic_range_compression(
            enhanced_signal,
            threshold=-24.0,
            ratio=4.0
        )

    return enhanced_signal

def add_frequency_boost(
    signal: np.ndarray,
    sample_rate: int,
    freq_range: Tuple[int, int],
    gain: float
) -> np.ndarray:
    """Apply frequency-specific boost to signal."""
    fft_out = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    fft_out[mask] *= gain

    return np.real(np.fft.ifft(fft_out))

def dynamic_range_compression(
    signal: np.ndarray,
    threshold: float,
    ratio: float
) -> np.ndarray:
    """Apply dynamic range compression to signal."""
    envelope = np.abs(signal)
    compressed = signal.copy()

    # Calculate gain based on compression parameters
    linear_gain = 1 / (1 + np.exp(-envelope/threshold))
    actual_ratio = np.maximum(ratio - 1, 1 - linear_gain * ratio)

    compressed = signal * actual_ratio
    return compressed

def process_audio(
    input_path: str,
    output_path: str,
    sr: Optional[int] = None,
    bass_boost: bool = True,
    treble_boost: bool = True,
    compression: bool = True
):
    """Main function to process audio file."""
    signal, original_sr = librosa.load(input_path, sr=sr)

    # Enhance quality
    enhanced_signal = enhance_quality(
        signal,
        sample_rate=original_sr,
        bass_boost=bass_boost,
        treble_boost=treble_boost,
        compression=compression
    )

    # Save processed audio
    sf.write(output_path, enhanced_signal, original_sr)

def main():
    if len(sys.argv) < 3:
        print("Usage: python audio_enhance.py <input_file> <output_file> [options]")
        print("Options:")
        print("  --no-bass-boost    Disable bass boost")
        print("  --no-treble-boost  Disable treble boost")
        print("  --no-compression   Disable compression")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse optional arguments
    bass_boost = True
    treble_boost = True
    compression = True

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--no-bass-boost":
            bass_boost = False
        elif sys.argv[i] == "--no-treble-boost":
            treble_boost = False
        elif sys.argv[i] == "--no-compression":
            compression = False

    process_audio(input_file, output_file,
                 bass_boost=bass_boost,
                 treble_boost=treble_boost,
                 compression=compression)

if __name__ == "__main__":
    main()
