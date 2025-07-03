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
from collections import deque
import threading
from queue import Queue
import re
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available - using fallback wakeword detection")

try:
    from scipy.signal import correlate
    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False

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

# Enhanced states for faster processing
LISTENING = "listening"
WAKEWORD_DETECTED = "wakeword_detected"
RECORDING_QUESTION = "recording_question"
PROCESSING_QUESTION = "processing_question"

# Global variables
current_state = LISTENING
recording = []
is_recording = False
silence_start = None
recording_start_time = None
is_transcribing = False

# Fast wakeword detection variables
audio_buffer = deque(maxlen=48000)  # 3 seconds at 16kHz
wakeword_buffer = deque(maxlen=64000)  # 4 seconds for wakeword detection
processing_queue = Queue()
wakeword_thread = None
wakeword_thread_running = False


def signal_handler(sig, frame):
    global shutdown_flag, wakeword_thread_running
    print("\n🛑 Ctrl+C pressed. Shutting down gracefully...")
    shutdown_flag = True
    wakeword_thread_running = False


signal.signal(signal.SIGINT, signal_handler)

# CLI Arguments
parser = argparse.ArgumentParser(
    description="🎤 Fast voice-activated recorder with 'hey oppenheimer' detection")
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
parser.add_argument("--fast_mode", action="store_true",
                    help="Enable ultra-fast wakeword detection")

args = parser.parse_args()

# Configuration
sample_rate = 16000
threshold = args.threshold
silence_duration = args.silence
max_record_seconds = args.max_time
api_base_url = args.api_endpoint
character_id = args.character_id
min_recording_duration = 0.5
fast_mode = args.fast_mode

# Enhanced wakeword patterns with phonetic variations
WAKEWORD_PATTERNS = [
    "hey oppenheimer", "oppenheimer", "hey openheimer", "openheimer",
    "hey oppenheim", "oppenheim", "hey oppenhimer", "oppenhimer",
    "hey oppenheimr", "oppenheimr", "hey oppenheime", "oppenheime"
]

# Phonetic pattern matching for better detection
PHONETIC_PATTERNS = [
    r"h[ae]y?\s*[ao]p+[ae]nh?[ae]im[ae]r?",
    r"[ao]p+[ae]nh?[ae]im[ae]r?",
    r"h[ae]y?\s*[ao]p+[ae]n[h]?[ae]im",
    r"[ao]p+[ae]n[h]?[ae]im"
]

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
lightweight_model = None

print(f"📦 Loading {args.model} model...")

try:
    if args.model == "whisper":
        import whisper
        whisper_model = whisper.load_model(args.whisper_size)
        print(f"✅ Whisper {args.whisper_size} loaded!")
        
        # Load lightweight model for fast wakeword detection
        if fast_mode:
            try:
                lightweight_model = whisper.load_model("tiny")
                print("✅ Lightweight Whisper tiny model loaded for fast detection!")
            except:
                print("⚠️ Could not load lightweight model, using main model")
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

# Fast wakeword detection functions


def compute_audio_features(audio_data):
    """Compute fast audio features for wakeword detection"""
    try:
        if len(audio_data) < 1000:
            return None
            
        if LIBROSA_AVAILABLE:
            # Compute MFCC features (fast)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # Compute spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            
            # Compute energy features
            rms = librosa.feature.rms(y=audio_data)
            
            return {
                'mfccs': mfccs,
                'spectral_centroids': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'rms': rms
            }
        else:
            # Fallback: basic energy features
            rms = np.sqrt(np.mean(audio_data**2))
            return {
                'rms': rms,
                'energy': np.sum(audio_data**2),
                'zero_crossing_rate': np.sum(np.diff(np.sign(audio_data)) != 0)
            }
    except Exception as e:
        print(f"⚠️ Feature computation error: {e}")
        return None


def fast_wakeword_check(audio_data):
    """Fast heuristic-based wakeword detection"""
    try:
        # Check audio length and energy
        if len(audio_data) < 8000:  # Less than 0.5 seconds
            return False
            
        # Compute RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < threshold * 0.5:
            return False
            
        # Check for speech-like patterns
        # Look for energy patterns that match "hey oppenheimer"
        # This is a simple heuristic based on syllable patterns
        
        # Divide audio into segments
        segment_length = len(audio_data) // 8
        segments = [audio_data[i:i+segment_length] for i in range(0, len(audio_data), segment_length)]
        
        # Compute energy for each segment
        energies = [np.sqrt(np.mean(seg**2)) for seg in segments if len(seg) > 0]
        
        if len(energies) < 4:
            return False
            
        # Look for energy patterns that could match "hey op-pen-hei-mer"
        # This is a very rough heuristic
        max_energy = max(energies)
        if max_energy < threshold:
            return False
            
        # Check for at least 2 peaks (syllables)
        peaks = sum(1 for e in energies if e > max_energy * 0.3)
        
        return peaks >= 2
        
    except Exception as e:
        print(f"⚠️ Fast wakeword check error: {e}")
        return False


def enhanced_wakeword_check(text):
    """Enhanced wakeword detection with phonetic matching"""
    if not text:
        return False, None
        
    text_lower = text.lower().strip()
    
    # Remove punctuation and extra spaces
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean)
    
    # Direct pattern matching
    for pattern in WAKEWORD_PATTERNS:
        if pattern in text_clean:
            return True, pattern
    
    # Phonetic pattern matching
    for pattern in PHONETIC_PATTERNS:
        if re.search(pattern, text_clean):
            return True, f"phonetic_match: {pattern}"
    
    # Fuzzy matching for common misrecognitions
    words = text_clean.split()
    if len(words) >= 1:
        # Check for variations of "oppenheimer"
        for word in words:
            if len(word) >= 6:
                # Check similarity to "oppenheimer"
                if (word.startswith('op') and 'heim' in word) or \
                   (word.startswith('ap') and 'heim' in word) or \
                   ('open' in word and 'heim' in word):
                    return True, f"fuzzy_match: {word}"
    
    return False, None


def lightweight_transcribe(audio_data):
    """Fast transcription using lightweight model"""
    try:
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Use lightweight model if available
        model_to_use = lightweight_model if lightweight_model else whisper_model
        
        if args.model == "whisper":
            # Fast transcription with minimal processing
            result = model_to_use.transcribe(
                audio_data, 
                language="en", 
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            return result["text"].strip()
        else:
            # Use the regular model for non-whisper
            return transcribe_audio_from_array(audio_data)
            
    except Exception as e:
        print(f"⚠️ Lightweight transcription error: {e}")
        return ""


def transcribe_audio_from_array(audio_data):
    """Transcribe audio from numpy array"""
    try:
        if args.model == "whisper":
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            result = whisper_model.transcribe(
                audio_data, language="en", temperature=0.0)
            return result["text"].strip()
        else:
            # For wav2vec2/hubert, ensure proper format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            input_values = processor(
                audio_data, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            return processor.decode(predicted_ids[0]).strip()
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


def transcribe_audio(audio_path):
    """Transcribe audio file (legacy function for compatibility)"""
    try:
        speech, sr = sf.read(audio_path)
        if sr != 16000:
            from scipy import signal as scipy_signal
            speech = scipy_signal.resample(
                speech, int(len(speech) * 16000 / sr))
        return transcribe_audio_from_array(speech)
    except Exception as e:
        print(f"❌ File transcription error: {e}")
        return ""


def wakeword_detection_thread():
    """Background thread for continuous wakeword detection"""
    global wakeword_thread_running, current_state
    
    print("🧵 Wakeword detection thread started")
    
    # Performance tracking
    last_detection_time = 0
    detection_cooldown = 2.0  # 2 second cooldown between detections
    
    while wakeword_thread_running and not shutdown_flag:
        try:
            # Check if we have enough audio data
            if len(wakeword_buffer) < 16000:  # Less than 1 second
                time.sleep(0.1)
                continue
            
            # Only process if we're listening
            if current_state != LISTENING:
                time.sleep(0.1)
                continue
            
            # Cooldown check to prevent rapid-fire detections
            current_time = time.time()
            if current_time - last_detection_time < detection_cooldown:
                time.sleep(0.1)
                continue
            
            # Get recent audio data (last 3 seconds)
            audio_data = np.array(list(wakeword_buffer)[-48000:])
            
            # Stage 1: Fast heuristic check
            if not fast_wakeword_check(audio_data):
                time.sleep(0.1)
                continue
            
            print("🔍 Stage 1 passed - running lightweight transcription")
            
            # Stage 2: Lightweight transcription
            transcript = lightweight_transcribe(audio_data)
            
            if transcript:
                print(f"📝 Lightweight transcript: '{transcript}'")
                
                # Stage 3: Enhanced wakeword detection
                wakeword_detected, pattern = enhanced_wakeword_check(transcript)
                
                if wakeword_detected:
                    print(f"🎯 WAKEWORD DETECTED: '{pattern}' in '{transcript}'")
                    current_state = WAKEWORD_DETECTED
                    wakeword_buffer.clear()  # Clear buffer after detection
                    last_detection_time = current_time
                    time.sleep(0.3)  # Brief pause before question recording
                    continue
            
            # Brief sleep to prevent excessive CPU usage
            time.sleep(0.05)
            
        except Exception as e:
            print(f"⚠️ Wakeword thread error: {e}")
            time.sleep(0.2)
    
    print("🧵 Wakeword detection thread stopped")


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
    """Enhanced audio callback function with real-time processing"""
    global is_recording, silence_start, recording, recording_start_time, current_state

    try:
        audio_chunk = indata.copy().flatten()
        volume_rms = np.sqrt(np.mean(audio_chunk**2))

        # Always add to wakeword buffer for continuous detection
        wakeword_buffer.extend(audio_chunk)

        if current_state == LISTENING:
            # Continuous listening - wakeword detection happens in background thread (fast mode)
            # or check periodically (legacy mode)
            if not fast_mode and len(wakeword_buffer) >= 32000:  # 2 seconds of audio
                # Run legacy detection every 2 seconds of audio
                try:
                    legacy_wakeword_detection()
                except Exception as e:
                    print(f"⚠️ Legacy wakeword detection error: {e}")

        elif current_state == WAKEWORD_DETECTED:
            # Wakeword detected, start recording question
            print("🎤 Starting question recording after wakeword detection")
            current_state = RECORDING_QUESTION
            recording = [audio_chunk]
            is_recording = True
            recording_start_time = time.time()
            silence_start = None

        elif current_state == RECORDING_QUESTION:
            # Record the question after wakeword detected
            if is_recording:
                recording.append(audio_chunk)

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

        elif current_state == PROCESSING_QUESTION:
            # Don't record while processing
            pass

    except Exception as e:
        print(f"❌ Callback error: {e}")


# Start wakeword detection thread
def start_wakeword_thread():
    global wakeword_thread, wakeword_thread_running
    
    if fast_mode:
        wakeword_thread_running = True
        wakeword_thread = threading.Thread(target=wakeword_detection_thread, daemon=True)
        wakeword_thread.start()
        print("🚀 Fast wakeword detection enabled")
    else:
        print("🐌 Using legacy wakeword detection - add --fast_mode for better performance")


def legacy_wakeword_detection():
    """Legacy wakeword detection for when fast mode is disabled"""
    global current_state, recording, is_recording, recording_start_time, silence_start
    
    # Check if we have enough audio for wakeword detection
    if len(wakeword_buffer) < 16000:  # Less than 1 second
        return False
    
    # Get recent audio data
    audio_data = np.array(list(wakeword_buffer))
    
    # Quick energy check
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < threshold * 0.3:
        return False
    
    # Use lightweight transcription
    transcript = lightweight_transcribe(audio_data)
    
    if transcript:
        print(f"📝 Legacy transcript: '{transcript}'")
        wakeword_detected, pattern = enhanced_wakeword_check(transcript)
        
        if wakeword_detected:
            print(f"🎯 WAKEWORD DETECTED (legacy): '{pattern}' in '{transcript}'")
            current_state = WAKEWORD_DETECTED
            wakeword_buffer.clear()
            return True
    
    return False


# Main execution
print("🎯 Starting enhanced voice assistant with fast 'hey oppenheimer' detection")
print(f"🔊 Volume threshold: {threshold}")
print(f"🤖 Using {args.model} model")
print(f"⚡ Fast mode: {'enabled' if fast_mode else 'disabled'}")
if fast_mode:
    print("🚀 Multi-stage detection: Heuristics → Lightweight transcription → Enhanced matching")
    print("⚡ Expected detection time: 0.1-0.5 seconds")
else:
    print("🐌 Legacy mode: Periodic transcription-based detection")
    print("⚡ Expected detection time: 1-3 seconds")
print("🎧 Say 'hey oppenheimer' to activate, then ask your question")
print("📝 Supported wake phrases:")
for pattern in WAKEWORD_PATTERNS[:4]:  # Show first 4 patterns
    print(f"   • '{pattern}'")
print("Press Ctrl+C to quit.")
print()

# Start the wakeword detection thread
start_wakeword_thread()

start_session = time.time()

try:
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
        while not shutdown_flag:
            time.sleep(0.1)

            # Process question (only when not using fast mode background processing)
            if current_state == PROCESSING_QUESTION and not is_recording and recording and not is_transcribing:
                is_transcribing = True

                try:
                    # Process the question
                    audio_data = np.concatenate(recording, axis=0)
                    
                    print(f"❓ Processing question ({len(audio_data)/sample_rate:.1f}s of audio)")
                    
                    # Transcribe the question
                    transcript = transcribe_audio_from_array(audio_data)
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
    wakeword_thread_running = False
    if wakeword_thread:
        wakeword_thread.join(timeout=2)
    try:
        pygame.mixer.quit()
    except:
        pass
    print("✅ Program ended.")
