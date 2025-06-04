"""
Speech-to-Text transcription module supporting multiple models.

This module provides voice-activated recording and transcription using
Whisper, Wav2Vec2, or HuBERT models.
"""

import os
import time
import tempfile
from typing import Dict, Any, Optional, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

# Set environment variables for stability
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass


def _load_whisper_model(model_size: str = "base"):
    """Load Whisper model."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "whisper is required for Whisper model support.\n"
            "Install with: pip install openai-whisper"
        )

    try:
        model = whisper.load_model(model_size)
        return model, None  # No processor needed for Whisper
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model: {e}")


def _load_wav2vec_model(model_name: str = "facebook/wav2vec2-large-960h"):
    """Load Wav2Vec2 model."""
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for Wav2Vec2 support.\n"
            "Install with: pip install transformers torch"
        )

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model = model.to('cpu')
        model.eval()
        return model, processor
    except Exception as e:
        raise TranscriptionError(f"Failed to load Wav2Vec2 model: {e}")


def _load_hubert_model(model_name: str = "facebook/hubert-large-ls960-ft"):
    """Load HuBERT model."""
    try:
        from transformers import Wav2Vec2Processor, HubertForCTC
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for HuBERT support.\n"
            "Install with: pip install transformers torch"
        )

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = HubertForCTC.from_pretrained(model_name)
        model = model.to('cpu')
        model.eval()
        return model, processor
    except Exception as e:
        raise TranscriptionError(f"Failed to load HuBERT model: {e}")


def _transcribe_whisper(model, audio_path: str, language: Optional[str] = "en") -> str:
    """Transcribe audio using Whisper."""
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            temperature=0.0,
            word_timestamps=True
        )
        transcript = result["text"].strip()
        return transcript if transcript else "No speech detected"
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}")


def _transcribe_wav2vec_hubert(model, processor, audio_path: str) -> str:
    """Transcribe audio using Wav2Vec2 or HuBERT."""
    try:
        import torch

        # Load and preprocess audio
        speech, sr = sf.read(audio_path)

        # Resample if needed
        if sr != 16000:
            from scipy import signal as scipy_signal
            speech = scipy_signal.resample(
                speech, int(len(speech) * 16000 / sr))
            sr = 16000

        # Ensure mono
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)

        # Normalize
        if np.max(np.abs(speech)) > 0:
            speech = speech / np.max(np.abs(speech)) * 0.8

        # Process with model
        input_values = processor(
            speech, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.decode(predicted_ids[0]).strip()
        return transcription if transcription else "No speech detected"

    except Exception as e:
        raise TranscriptionError(f"Model transcription failed: {e}")


class VoiceRecorder:
    """Voice-activated audio recorder."""

    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.05)
        self.silence_duration = config.get('silence', 2.0)
        self.max_time = config.get('max_time', 30)
        self.sample_rate = config.get('sample_rate', 16000)
        self.min_duration = config.get('min_duration', 0.5)

        # Recording state
        self.recording = []
        self.is_recording = False
        self.silence_start = None
        self.recording_start_time = None
        self.finished = False
        self.start_time = time.time()

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for recording."""
        try:
            # Calculate volume metrics
            volume_rms = np.sqrt(np.mean(indata**2))
            current_volume = volume_rms

            # Voice activity detection
            voice_detected = current_volume > self.threshold

            # Additional noise filtering
            if voice_detected and len(self.recording) > 0:
                recent_volumes = [np.sqrt(np.mean(chunk**2))
                                  for chunk in self.recording[-5:]]
                if len(recent_volumes) >= 3:
                    avg_recent_volume = np.mean(recent_volumes)
                    if avg_recent_volume < self.threshold * 0.3:
                        voice_detected = False

            if voice_detected:
                if not self.is_recording:
                    print(
                        f"🎙️ Recording started (volume: {current_volume:.4f})")
                    self.is_recording = True
                    self.recording_start_time = time.time()
                self.silence_start = None
                self.recording.append(indata.copy())

            elif self.is_recording:
                # Continue recording during brief pauses
                self.recording.append(indata.copy())

                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start > self.silence_duration:
                    recording_duration = time.time() - self.recording_start_time
                    if recording_duration >= self.min_duration:
                        print(
                            f"🛑 Recording finished (duration: {recording_duration:.1f}s)")
                        self.finished = True
                    else:
                        print(
                            f"⚠️ Recording too short ({recording_duration:.1f}s), continuing...")
                        self.recording = []
                        self.is_recording = False
                        self.silence_start = None

            # Check max time
            if time.time() - self.start_time > self.max_time:
                if self.recording:
                    print(f"⏱️ Max time reached, finishing recording")
                    self.finished = True
                else:
                    raise TranscriptionError(
                        "Max recording time reached with no audio")

        except Exception as e:
            print(f"❌ Audio callback error: {e}")

    def record(self) -> Optional[str]:
        """Record audio and return path to saved file."""
        print(
            f"🎧 Listening... (threshold: {self.threshold}, max: {self.max_time}s)")
        print("Speak to start recording...")

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback
            ):
                while not self.finished:
                    time.sleep(0.1)

            if not self.recording:
                return None

            # Process recorded audio
            print(f"📊 Processing {len(self.recording)} audio chunks...")
            audio_data = np.concatenate(self.recording, axis=0)

            # Ensure 1D audio
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8

            # Add padding
            silence_padding = np.zeros(
                self.sample_rate // 2, dtype=audio_data.dtype)
            padded_audio = np.concatenate(
                (silence_padding, audio_data, silence_padding))

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Save as 16-bit PCM WAV
            audio_int16 = (padded_audio * 32767).astype(np.int16)
            write(temp_filename, self.sample_rate, audio_int16)

            print(f"✅ Audio saved to {temp_filename}")
            return temp_filename

        except Exception as e:
            raise TranscriptionError(f"Recording failed: {e}")


def listen_and_transcribe(
    model: str,
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """
    Listen for audio input and transcribe using specified model.

    Args:
        model: Model type ('whisper', 'wav2vec', 'hubert')
        configuration: Optional configuration dict with parameters:
            - threshold: Volume threshold to start recording (default: 0.05)
            - silence: Seconds of silence before stopping (default: 2.0)
            - max_time: Maximum recording time in seconds (default: 30)
            - sample_rate: Audio sample rate (default: 16000)
            - min_duration: Minimum recording duration (default: 0.5)
            - model_size: Model size for Whisper ('tiny', 'base', 'small', 'medium', 'large')
            - model_name: Custom model name for wav2vec/hubert
            - language: Language for Whisper (default: 'en')
            - output_dir: Directory to save recordings (optional)

    Returns:
        Transcribed text string

    Raises:
        TranscriptionError: If transcription fails
        ImportError: If required packages are missing
        ValueError: If model type is not supported
    """
    if configuration is None:
        configuration = {}

    model = model.lower()
    if model not in ['whisper', 'wav2vec', 'hubert']:
        raise ValueError(
            f"Unsupported model: {model}. Use 'whisper', 'wav2vec', or 'hubert'")

    print(f"📦 Loading {model} model...")

    # Load appropriate model
    if model == 'whisper':
        model_size = configuration.get('model_size', 'base')
        loaded_model, processor = _load_whisper_model(model_size)
    elif model == 'wav2vec':
        model_name = configuration.get(
            'model_name', 'facebook/wav2vec2-large-960h')
        loaded_model, processor = _load_wav2vec_model(model_name)
    else:  # hubert
        model_name = configuration.get(
            'model_name', 'facebook/hubert-large-ls960-ft')
        loaded_model, processor = _load_hubert_model(model_name)

    print("✅ Model loaded successfully!")

    # Record audio
    recorder = VoiceRecorder(configuration)
    audio_file = recorder.record()

    if not audio_file:
        return "No audio recorded"

    try:
        print("📝 Transcribing...")

        # Transcribe based on model type
        if model == 'whisper':
            language = configuration.get('language', 'en')
            transcript = _transcribe_whisper(
                loaded_model, audio_file, language)
        else:
            transcript = _transcribe_wav2vec_hubert(
                loaded_model, processor, audio_file)

        # Save recording if output directory specified
        output_dir = configuration.get('output_dir')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            final_path = os.path.join(
                output_dir, f"recording_{int(time.time())}.wav")
            os.rename(audio_file, final_path)
            print(f"📁 Recording saved to: {final_path}")
        else:
            # Clean up temporary file
            try:
                os.unlink(audio_file)
            except:
                pass

        print(f"📄 Transcript: '{transcript}'")
        return transcript

    except Exception as e:
        # Clean up temporary file on error
        try:
            os.unlink(audio_file)
        except:
            pass
        raise TranscriptionError(f"Transcription failed: {e}")


# Convenience functions for each model type
def listen_with_whisper(config: Optional[Dict[str, Any]] = None) -> str:
    """Listen and transcribe using Whisper."""
    return listen_and_transcribe('whisper', config)


def listen_with_wav2vec(config: Optional[Dict[str, Any]] = None) -> str:
    """Listen and transcribe using Wav2Vec2."""
    return listen_and_transcribe('wav2vec', config)


def listen_with_hubert(config: Optional[Dict[str, Any]] = None) -> str:
    """Listen and transcribe using HuBERT."""
    return listen_and_transcribe('hubert', config)


# Convenience functions for transcribing audio files
def transcribe_with_whisper(audio_file_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Transcribe audio file using Whisper."""
    return transcribe_audio_file(audio_file_path, 'whisper', config)


def transcribe_with_wav2vec(audio_file_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Transcribe audio file using Wav2Vec2."""
    return transcribe_audio_file(audio_file_path, 'wav2vec', config)


def transcribe_with_hubert(audio_file_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Transcribe audio file using HuBERT."""
    return transcribe_audio_file(audio_file_path, 'hubert', config)


def transcribe_audio_file(
    audio_file_path: str,
    model: str,
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """
    Transcribe a pre-recorded audio file using specified model.

    Args:
        audio_file_path: Path to the audio file to transcribe
        model: Model type ('whisper', 'wav2vec', 'hubert')
        configuration: Optional configuration dict with parameters:
            - model_size: Model size for Whisper ('tiny', 'base', 'small', 'medium', 'large')
            - model_name: Custom model name for wav2vec/hubert
            - language: Language for Whisper (default: 'en')

    Returns:
        Transcribed text string

    Raises:
        TranscriptionError: If transcription fails
        ImportError: If required packages are missing
        ValueError: If model type is not supported
        FileNotFoundError: If audio file doesn't exist
    """
    if configuration is None:
        configuration = {}

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    model = model.lower()
    if model not in ['whisper', 'wav2vec', 'hubert']:
        raise ValueError(
            f"Unsupported model: {model}. Use 'whisper', 'wav2vec', or 'hubert'")

    print(f"📦 Loading {model} model...")

    # Load appropriate model
    if model == 'whisper':
        model_size = configuration.get('model_size', 'base')
        loaded_model, processor = _load_whisper_model(model_size)
    elif model == 'wav2vec':
        model_name = configuration.get(
            'model_name', 'facebook/wav2vec2-large-960h')
        loaded_model, processor = _load_wav2vec_model(model_name)
    else:  # hubert
        model_name = configuration.get(
            'model_name', 'facebook/hubert-large-ls960-ft')
        loaded_model, processor = _load_hubert_model(model_name)

    print("✅ Model loaded successfully!")

    try:
        print("📝 Transcribing audio file...")

        # Transcribe based on model type
        if model == 'whisper':
            language = configuration.get('language', 'en')
            transcript = _transcribe_whisper(
                loaded_model, audio_file_path, language)
        else:
            transcript = _transcribe_wav2vec_hubert(
                loaded_model, processor, audio_file_path)

        print(f"📄 Transcript: '{transcript}'")
        return transcript

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")
