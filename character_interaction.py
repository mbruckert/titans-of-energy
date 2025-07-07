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
import random
from scipy.io.wavfile import write
from collections import deque
import threading
import queue

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
RECORDING_QUESTION = "recording_question"
PROCESSING_QUESTION = "processing_question"

# Global variables
current_state = LISTENING
recording = []
is_recording = False
silence_start = None
recording_start_time = None
audio_buffer = deque(maxlen=48000)  # 3 seconds at 16kHz
wakeword_queue = queue.Queue()
transcription_thread = None
last_wakeword_detection = 0  # Timestamp of last wakeword detection
wakeword_cooldown = 3.0  # Cooldown period in seconds

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
                    "medium", "large", "large-v2"], default="base", help="Whisper model size")
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
parser.add_argument("--wakeword_threshold", type=float,
                    default=0.01, help="Volume threshold for wakeword detection")

args = parser.parse_args()

# Configuration
sample_rate = 16000
threshold = args.threshold
wakeword_threshold = args.wakeword_threshold
silence_duration = args.silence
max_record_seconds = args.max_time
api_base_url = args.api_endpoint
character_id = args.character_id
min_recording_duration = 0.5

# Wakeword patterns - will be populated dynamically based on character
WAKEWORD_PATTERNS = []

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
    """Check if wakeword is in text with fuzzy matching"""
    text_lower = text.lower().strip()
    
    # Direct pattern matching
    for pattern in WAKEWORD_PATTERNS:
        if pattern in text_lower:
            return True, pattern
    
    # Fuzzy matching for common transcription errors
    words = text_lower.split()
    if len(words) >= 1:
        # Check for "oppenheimer" variations
        for word in words:
            if ("oppen" in word and len(word) > 4) or \
               ("open" in word and "heim" in word) or \
               (word.startswith("op") and "heim" in word):
                return True, word
        
        # Check for "hey" + name combinations
        for i in range(len(words) - 1):
            if words[i] in ["hey", "hi", "hello"] and \
               ("oppen" in words[i+1] or "open" in words[i+1]):
                return True, f"{words[i]} {words[i+1]}"
    
    return False, None

def transcribe_audio_optimized(audio_path):
    """Transcribe audio file with optimized settings for wakeword detection"""
    try:
        if args.model == "whisper":
            # Use more aggressive settings for better accuracy
            result = whisper_model.transcribe(
                audio_path, 
                language="en", 
                temperature=0.0,
                best_of=2,  # Try multiple attempts
                beam_size=5,  # Better beam search
                word_timestamps=False,
                condition_on_previous_text=False
            )
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

def wakeword_detection_worker():
    """Background thread for processing wakeword detection"""
    global current_state, recording, is_recording, recording_start_time, silence_start, last_wakeword_detection
    
    while not shutdown_flag:
        try:
            # Get audio data from queue with timeout
            audio_data = wakeword_queue.get(timeout=0.1)
            
            if audio_data is None:  # Shutdown signal
                break
                
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                filename = temp_file.name
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8
            
            audio_int16 = (audio_data * 32767).astype(np.int16)
            write(filename, sample_rate, audio_int16)
            
            # Transcribe
            transcript = transcribe_audio_optimized(filename)
            
            # Clean up temp file
            try:
                os.unlink(filename)
            except:
                pass
            
            if transcript:
                print(f"🎯 Checking: '{transcript}'")
                
                # Check cooldown period
                current_time = time.time()
                if current_time - last_wakeword_detection < wakeword_cooldown:
                    print(f"⏰ Wakeword cooldown active ({wakeword_cooldown - (current_time - last_wakeword_detection):.1f}s remaining)")
                    continue
                
                # Check for wakeword
                wakeword_detected, pattern = check_for_wakeword(transcript)
                
                if wakeword_detected:
                    print(f"✅ Wakeword '{pattern}' detected! Now listening for your question...")
                    
                    # Update last detection timestamp
                    last_wakeword_detection = current_time
                    
                    # Play start recording sound
                    play_notification_sound("start")
                    
                    current_state = RECORDING_QUESTION
                    recording = []
                    is_recording = True
                    recording_start_time = time.time()
                    silence_start = None
                    
                    print(f"🎤 Started recording question (threshold: {max(threshold * 2, 0.01):.4f})")
                    print(f"🔇 Will stop after {silence_duration}s of silence")
                    
                    # Clear the wakeword queue to prevent duplicate detections
                    try:
                        while not wakeword_queue.empty():
                            wakeword_queue.get_nowait()
                            wakeword_queue.task_done()
                    except queue.Empty:
                        pass
                else:
                    print(f"❌ No wakeword in: '{transcript}'")
            
            wakeword_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️ Wakeword detection error: {e}")

def fetch_character_wakeword(character_id):
    """Fetch character's wakeword from API"""
    try:
        response = requests.get(f"{api_base_url}/get-character-wakeword/{character_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data.get('wakeword', 'hey character')
        return None
    except Exception as e:
        print(f"❌ Error fetching wakeword: {e}")
        return None

def setup_wakeword_patterns(base_wakeword):
    """Setup wakeword patterns based on character's wakeword"""
    global WAKEWORD_PATTERNS
    
    if not base_wakeword:
        base_wakeword = "hey character"
    
    base_lower = base_wakeword.lower()
    
    # Generate variations for better matching
    patterns = [base_lower]
    
    # Add variations with different greetings
    if base_lower.startswith("hey "):
        name_part = base_lower[4:]  # Remove "hey "
        patterns.extend([
            f"hi {name_part}",
            f"hello {name_part}",
            name_part,  # Just the name
            f"hey {name_part}",  # Original
        ])
    else:
        # If it doesn't start with "hey", add greeting variations
        patterns.extend([
            f"hey {base_lower}",
            f"hi {base_lower}",
            f"hello {base_lower}",
        ])
    
    # Add common transcription errors/variations
    for pattern in patterns.copy():
        if "oppenheimer" in pattern:
            patterns.extend([
                pattern.replace("oppenheimer", "openheimer"),
                pattern.replace("oppenheimer", "oppenheimmer"),
                pattern.replace("oppenheimer", "oppenhiemer"),
                pattern.replace("oppenheimer", "openhimer"),
            ])
    
    WAKEWORD_PATTERNS = list(set(patterns))  # Remove duplicates
    print(f"🎯 Wakeword patterns: {WAKEWORD_PATTERNS}")

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

def play_thinking_audio():
    """Play a random thinking audio clip from character's API"""
    try:
        print("🤔 Fetching thinking audio from API...")
        response = requests.get(f"{api_base_url}/get-character-thinking-audio/{character_id}", timeout=10)
        
        if response.status_code != 200:
            print("⚠️ No thinking audio available for this character")
            return False
            
        data = response.json()
        
        if data.get('status') != 'success' or not data.get('audio_base64'):
            print("⚠️ No thinking audio data received")
            return False
        
        # Convert base64 to audio data and play
        audio_data = base64_to_audio_data(data['audio_base64'])
        if audio_data:
            print(f"🤔 Playing thinking audio (phrase: {data.get('phrase_id', 'unknown')})")
            return play_audio_response(audio_data)
        else:
            print("❌ Failed to decode thinking audio")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching/playing thinking audio: {e}")
        return False

def play_notification_sound(sound_type="start"):
    """Play a notification sound for recording events"""
    try:
        # Generate different tones for start and stop
        if sound_type == "start":
            # Rising tone for recording start
            duration = 0.15  # seconds
            sample_rate = 44100
            frequency1 = 800  # Hz
            frequency2 = 1200  # Hz
        else:  # stop
            # Falling tone for recording stop
            duration = 0.15  # seconds
            sample_rate = 44100
            frequency1 = 1200  # Hz
            frequency2 = 800  # Hz
        
        # Generate tone
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Create a frequency sweep from frequency1 to frequency2
        frequencies = np.linspace(frequency1, frequency2, len(t))
        wave = np.sin(2 * np.pi * frequencies * t) * 0.3  # Lower volume
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.05 * sample_rate)  # 50ms fade
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Convert to 16-bit integers
        wave_int16 = (wave * 32767).astype(np.int16)
        
        # Create stereo audio (duplicate mono to both channels)
        stereo_wave = np.column_stack((wave_int16, wave_int16))
        
        # Play using pygame
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)
        sound_array = pygame.sndarray.make_sound(stereo_wave)
        sound_array.play()
        
        # Wait for sound to finish
        time.sleep(duration + 0.1)
        
        return True
    except Exception as e:
        print(f"❌ Error playing notification sound: {e}")
        return False

def callback(indata, frames, time_info, status):
    """Audio callback function"""
    global is_recording, silence_start, recording, recording_start_time, current_state, audio_buffer

    try:
        audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        volume_rms = np.sqrt(np.mean(audio_data**2))

        if current_state == LISTENING:
            # Continuously buffer audio for wakeword detection
            audio_buffer.extend(audio_data)
            
            # Check for speech above wakeword threshold
            if volume_rms > wakeword_threshold:
                # Get buffered audio (last 3 seconds)
                if len(audio_buffer) >= 16000:  # At least 1 second
                    buffered_audio = np.array(audio_buffer)
                    
                    # Add to wakeword detection queue (non-blocking)
                    try:
                        wakeword_queue.put_nowait(buffered_audio.copy())
                    except queue.Full:
                        pass  # Skip if queue is full
                    
                    # Clear some of the buffer to avoid overlap
                    for _ in range(8000):  # Remove 0.5 seconds
                        if audio_buffer:
                            audio_buffer.popleft()

        elif current_state == RECORDING_QUESTION:
            # Clear audio buffer when first entering recording state to avoid wake word contamination
            if len(audio_buffer) > 0:
                audio_buffer.clear()
            
            # Record the question after wakeword detected
            if is_recording:
                recording.append(audio_data.copy())

                # Use a more reasonable silence threshold for question recording
                # Since threshold is for wakeword detection (0.02), use a higher threshold for speech
                speech_threshold = max(threshold * 2, 0.01)  # At least 0.01, or 2x wakeword threshold
                
                if volume_rms > speech_threshold:
                    silence_start = None
                    print(f"🎤 Recording... (volume: {volume_rms:.4f})")
                else:
                    if silence_start is None:
                        silence_start = time.time()
                        print(f"🔇 Silence detected, waiting {silence_duration}s to stop...")
                    elif time.time() - silence_start > silence_duration:
                        recording_duration = time.time() - recording_start_time
                        if recording_duration >= min_recording_duration:
                            print(f"✅ Recording stopped after {recording_duration:.1f}s")
                            is_recording = False
                            current_state = PROCESSING_QUESTION
                            # Play stop recording sound in a separate thread to avoid blocking
                            threading.Thread(target=lambda: play_notification_sound("stop"), daemon=True).start()
                        else:
                            print(f"⚠️ Recording too short ({recording_duration:.1f}s), continuing...")
                            silence_start = None

        elif current_state == PROCESSING_QUESTION:
            # Don't record while processing
            pass

    except Exception as e:
        print(f"❌ Callback error: {e}")

# Fetch character's wakeword and setup patterns
print(f"🔄 Fetching wakeword for character {character_id}...")
character_wakeword = fetch_character_wakeword(character_id)
if character_wakeword:
    print(f"✅ Character wakeword: '{character_wakeword}'")
    setup_wakeword_patterns(character_wakeword)
else:
    print("⚠️ Failed to fetch character wakeword, using default patterns")
    setup_wakeword_patterns("hey character")

# Start wakeword detection thread
transcription_thread = threading.Thread(target=wakeword_detection_worker, daemon=True)
transcription_thread.start()

# Main execution
print("🎯 Starting voice assistant with dynamic wakeword detection")
print(f"🔊 Volume threshold: {threshold}")
print(f"🎯 Wakeword threshold: {wakeword_threshold}")
print(f"🤖 Using {args.model} model ({args.whisper_size if args.model == 'whisper' else 'default'})")
print(f"🎧 Say '{character_wakeword or 'hey character'}' to activate, then ask your question")
print("Press Ctrl+C to quit.")

start_session = time.time()

try:
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
        while not shutdown_flag:
            time.sleep(0.1)

            # Process question
            if current_state == PROCESSING_QUESTION and not is_recording and recording:
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

                    transcript = transcribe_audio_optimized(filename)
                    print(f"❓ Question: '{transcript}'")

                    # Play thinking audio while processing
                    if transcript:
                        play_thinking_audio()
                        
                        # Call API and play response
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
                current_state = LISTENING
                # Clear audio buffer to start fresh
                audio_buffer.clear()
                print(f"🎧 Listening for '{character_wakeword or 'hey character'}' again...")

            # Session timeout
            if time.time() - start_session > max_record_seconds:
                print("⏱️ Session timeout. Exiting.")
                break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"❌ Main loop error: {e}")
finally:
    # Shutdown wakeword detection thread
    try:
        wakeword_queue.put(None)  # Shutdown signal
        transcription_thread.join(timeout=2)
    except:
        pass
    
    try:
        pygame.mixer.quit()
    except:
        pass
    print("✅ Program ended.")