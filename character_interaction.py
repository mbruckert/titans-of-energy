#!/usr/bin/env python3

import argparse
import os
import time
import signal
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import requests
import base64
import pygame
import io
import tempfile
import warnings
from scipy.io.wavfile import write

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="whisper.transcribe")

# Set environment variables to potentially fix bus errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Global flag for graceful shutdown
shutdown_flag = False

# Simple states
LISTENING = "listening"
CHECKING_WAKEWORD = "checking_wakeword"
RECORDING_QUESTION = "recording_question"
PROCESSING_WAKEWORD = "processing_wakeword"
PROCESSING_QUESTION = "processing_question"

# Global variables
current_state = LISTENING
recording = []
is_recording = False
silence_start = None
recording_start_time = None
is_transcribing = False


def signal_handler(sig, frame):
    global shutdown_flag
    print("\n🛑 Ctrl+C pressed. Shutting down gracefully...")
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)

# CLI Arguments
parser = argparse.ArgumentParser(
    description="🎤 Voice-activated recorder with 'hey oppenheimer' detection")
parser.add_argument(
    "--model", choices=["whisper", "wav2vec2", "hubert"], default="whisper", help="STT model")
parser.add_argument("--whisper_size", choices=["tiny", "base", "small",
                    "medium", "large", "large-v2"], default="tiny", help="Whisper model size")
parser.add_argument("--threshold", type=float,
                    default=0.02, help="Volume threshold")
parser.add_argument("--silence", type=float,
                    default=2.0, help="Silence duration")
parser.add_argument("--max_time", type=int, default=300,
                    help="Max session time")
parser.add_argument("--output", type=str, default=".", help="Output folder")
parser.add_argument("--api_endpoint", type=str,
                    default="http://localhost:5000", help="API endpoint")
parser.add_argument("--character_id", type=int,
                    required=True, help="Character ID")

args = parser.parse_args()

# Configuration
sample_rate = 16000
threshold = args.threshold
silence_duration = args.silence
max_record_seconds = args.max_time
api_base_url = args.api_endpoint
character_id = args.character_id
min_recording_duration = 0.5
wakeword_timeout = 4.0  # Max time to wait for wakeword

# Wakeword patterns
WAKEWORD_PATTERNS = ["hey oppenheimer",
                     "oppenheimer", "hey openheimer", "openheimer"]

# Initialize audio
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    print("🔊 Audio playback initialized")
except pygame.error as e:
    print(f"⚠️ Audio playback warning: {e}")

# Load models
model = None
processor = None
whisper_model = None

print(f"📦 Loading {args.model} model...")

try:
    if args.model == "whisper":
        import whisper
        whisper_model = whisper.load_model(args.whisper_size)
        print(f"✅ Whisper {args.whisper_size} loaded!")
    else:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
        hf_model_id = {"wav2vec2": "facebook/wav2vec2-large-960h",
                       "hubert": "facebook/hubert-large-ls960-ft"}[args.model]
        processor = Wav2Vec2Processor.from_pretrained(hf_model_id)
        if args.model == "wav2vec2":
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h")
        else:
            model = HubertForCTC.from_pretrained(
                "facebook/hubert-large-ls960-ft")
        model = model.to('cpu')
        model.eval()
        print(f"✅ {args.model} loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Helper functions


def check_for_wakeword(text):
    """Check if wakeword is in text"""
    text_lower = text.lower().strip()
    for pattern in WAKEWORD_PATTERNS:
        if pattern in text_lower:
            return True, pattern
    return False, None


def transcribe_audio(audio_path):
    """Transcribe audio file"""
    try:
        if args.model == "whisper":
            result = whisper_model.transcribe(
                audio_path, language="en", temperature=0.0)
            return result["text"].strip()
        else:
            speech, sr = sf.read(audio_path)
            if sr != 16000:
                from scipy import signal as scipy_signal
                speech = scipy_signal.resample(
                    speech, int(len(speech) * 16000 / sr))
            if len(speech.shape) > 1:
                speech = speech.mean(axis=1)
            if np.max(np.abs(speech)) > 0:
                speech = speech / np.max(np.abs(speech)) * 0.8
            input_values = processor(
                speech, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            return processor.decode(predicted_ids[0]).strip()
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


def call_character_api(question):
    """Call character API"""
    try:
        response = requests.post(f"{api_base_url}/ask-question-text", json={
                                 "character_id": character_id, "question": question}, timeout=60)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"❌ API error: {e}")
        return None


def play_audio_response(audio_data):
    """Play audio response"""
    try:
        audio_buffer = io.BytesIO(audio_data)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        return True
    except Exception as e:
        print(f"❌ Audio playback error: {e}")
        return False


def base64_to_audio_data(base64_string):
    """Convert base64 to audio data"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        print(f"❌ Base64 decode error: {e}")
        return None


def callback(indata, frames, time_info, status):
    """Audio callback function"""
    global is_recording, silence_start, recording, recording_start_time, current_state

    try:
        volume_rms = np.sqrt(np.mean(indata**2))

        if current_state == LISTENING:
            # Listen for speech above threshold
            if volume_rms > threshold:
                print(f"🎙️ Speech detected! Checking for wakeword...")
                current_state = CHECKING_WAKEWORD
                recording = [indata.copy()]
                is_recording = True
                recording_start_time = time.time()
                silence_start = None

        elif current_state == CHECKING_WAKEWORD:
            # Record audio to check for wakeword
            if is_recording:
                recording.append(indata.copy())

                # Check if we should stop recording (silence or timeout)
                if volume_rms > threshold * 0.3:  # Lower threshold for continuation
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > 1.5:  # Shorter silence for wakeword check
                        recording_duration = time.time() - recording_start_time
                        if recording_duration >= 0.8:  # At least 0.8 seconds
                            is_recording = False
                            current_state = PROCESSING_WAKEWORD
                        else:
                            # Too short, go back to listening
                            current_state = LISTENING
                            recording = []
                            is_recording = False

                # Timeout check
                if time.time() - recording_start_time > wakeword_timeout:
                    is_recording = False
                    current_state = PROCESSING_WAKEWORD

        elif current_state == RECORDING_QUESTION:
            # Record the question after wakeword detected
            if is_recording:
                recording.append(indata.copy())

                if volume_rms > threshold * 0.3:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        recording_duration = time.time() - recording_start_time
                        if recording_duration >= min_recording_duration:
                            is_recording = False
                            current_state = PROCESSING_QUESTION
                        else:
                            silence_start = None

        elif current_state in [PROCESSING_WAKEWORD, PROCESSING_QUESTION]:
            # Don't record while processing
            pass

    except Exception as e:
        print(f"❌ Callback error: {e}")


# Main execution
print("🎯 Starting voice assistant with 'hey oppenheimer' detection")
print(f"🔊 Volume threshold: {threshold}")
print(f"🤖 Using {args.model} model")
print("🎧 Say 'hey oppenheimer' to activate, then ask your question")
print("Press Ctrl+C to quit.")

start_session = time.time()

try:
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
        while not shutdown_flag:
            time.sleep(0.1)

            # Process wakeword check
            if current_state == PROCESSING_WAKEWORD and not is_recording and recording and not is_transcribing:
                is_transcribing = True

                try:
                    # Prepare audio data
                    audio_data = np.concatenate(recording, axis=0)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()

                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.8

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        filename = temp_file.name

                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    write(filename, sample_rate, audio_int16)

                    # Transcribe
                    transcript = transcribe_audio(filename)
                    print(f"📝 Transcribed: '{transcript}'")

                    # Check for wakeword
                    wakeword_detected, pattern = check_for_wakeword(transcript)

                    if wakeword_detected:
                        print(
                            f"🎯 Wakeword '{pattern}' detected! Now listening for your question...")
                        current_state = RECORDING_QUESTION
                        recording = []
                        is_recording = True
                        recording_start_time = time.time()
                        silence_start = None
                    else:
                        print(f"❌ No wakeword detected. Going back to listening...")
                        current_state = LISTENING

                    # Clean up
                    try:
                        os.unlink(filename)
                    except:
                        pass

                except Exception as e:
                    print(f"⚠️ Processing error: {e}")
                    current_state = LISTENING

                recording = []
                is_transcribing = False

            # Process question
            elif current_state == PROCESSING_QUESTION and not is_recording and recording and not is_transcribing:
                is_transcribing = True

                try:
                    # Process the question
                    audio_data = np.concatenate(recording, axis=0)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()

                    # Save and transcribe
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        filename = temp_file.name

                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.8

                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    write(filename, sample_rate, audio_int16)

                    transcript = transcribe_audio(filename)
                    print(f"❓ Question: '{transcript}'")

                    # Call API and play response
                    if transcript:
                        api_response = call_character_api(transcript)
                        if api_response and api_response.get('status') == 'success':
                            if 'text_response' in api_response:
                                print(
                                    f"💬 Response: '{api_response['text_response']}'")

                            if 'audio_base64' in api_response and api_response['audio_base64']:
                                audio_data = base64_to_audio_data(
                                    api_response['audio_base64'])
                                if audio_data:
                                    play_audio_response(audio_data)
                                    print("🎵 Response played.")

                    # Clean up
                    try:
                        os.unlink(filename)
                    except:
                        pass

                except Exception as e:
                    print(f"⚠️ Question processing error: {e}")

                # Reset to listening
                recording = []
                is_transcribing = False
                current_state = LISTENING
                print("🎧 Listening for 'hey oppenheimer' again...")

            # Session timeout
            if time.time() - start_session > max_record_seconds:
                print("⏱️ Session timeout. Exiting.")
                break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"❌ Main loop error: {e}")
finally:
    try:
        pygame.mixer.quit()
    except:
        pass
    print("✅ Program ended.")
