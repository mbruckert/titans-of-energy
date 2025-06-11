#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import signal
import sys
import argparse
import requests
import base64
import pygame
import io
import tempfile
import whisper  # pip install openai-whisper

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    global shutdown_flag
    print("\n🛑 Ctrl+C pressed. Shutting down gracefully...")
    shutdown_flag = True


# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

# ---------------- CLI Arguments ----------------
parser = argparse.ArgumentParser(
    description="🎤 Voice-activated recorder and transcriber using Whisper")

parser.add_argument("--threshold", type=float, default=0.05,
                    help="Volume threshold to start recording")
parser.add_argument("--silence", type=float, default=2.0,
                    help="Seconds of silence before stopping")
parser.add_argument("--max_time", type=int, default=300,
                    help="Max session time in seconds")
parser.add_argument("--output", type=str, default=".",
                    help="Output folder for recordings")
parser.add_argument("--api_endpoint", type=str, default="http://localhost:5000",
                    help="Base URL for the character API")
parser.add_argument("--character_id", type=int, required=True,
                    help="Character ID to use for responses")
parser.add_argument("--whisper_model", type=str, default="small",
                    choices=["tiny", "base", "small",
                             "medium", "large", "large-v2"],
                    help="Whisper model size to use")

args = parser.parse_args()

# ---------------- Configuration ----------------
sample_rate = 16000
threshold = args.threshold
silence_duration = args.silence
max_record_seconds = args.max_time
output_folder = args.output
api_base_url = args.api_endpoint
character_id = args.character_id
whisper_model_size = args.whisper_model
min_recording_duration = 0.5  # Minimum recording duration in seconds

recording = []
is_recording = False
is_transcribing = False
is_processing_response = False
silence_start = None
recording_start_time = None

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    print("🔊 Audio playback initialized")
except pygame.error as e:
    print(f"⚠️ Warning: Could not initialize pygame mixer: {e}")
    print("Audio playback may not work properly.")

# Load Whisper model
print(f"📦 Loading Whisper model: {whisper_model_size}...")
try:
    model = whisper.load_model(whisper_model_size)
    print("✅ Whisper model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    sys.exit(1)


# ---------------- Helper Functions ----------------

def call_character_api(question):
    """Call the character API endpoint with the transcribed question"""
    print("🌐 Calling character API endpoint...")
    try:
        url = f"{api_base_url}/ask-question-text"
        payload = {"character_id": character_id, "question": question}

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling API endpoint: {e}")
        return None


def base64_to_audio_data(base64_string):
    """Convert base64 string to audio data"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_string)
        return audio_bytes
    except Exception as e:
        print(f"❌ Error converting base64 to audio: {e}")
        return None


def play_audio_response(audio_data):
    """Play audio response using pygame with fallback options"""
    print("🔊 Playing audio response...")
    try:
        # Create a BytesIO object from the audio data
        audio_buffer = io.BytesIO(audio_data)

        # Try to load and play the audio
        try:
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            print("✅ Audio playback completed.")
            return True

        except pygame.error as e:
            print(f"⚠️ Pygame playback failed: {e}")
            # Try saving to temp file and playing with pygame
            return play_audio_fallback(audio_data)

    except Exception as e:
        print(f"❌ Error playing audio: {e}")
        return play_audio_fallback(audio_data)


def play_audio_fallback(audio_data):
    """Fallback audio playback method using temporary file"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_data)

        # Try playing with pygame from file
        try:
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            print("✅ Audio playback completed (fallback method).")
            return True

        except pygame.error:
            print(
                "❌ Pygame fallback also failed. Audio response received but cannot be played.")
            return False

    except Exception as e:
        print(f"❌ Fallback audio playback failed: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            if 'temp_filename' in locals():
                os.unlink(temp_filename)
        except:
            pass


def play_thinking_audio():
    """Play thinking audio while waiting for API response"""
    try:
        # Look for thinking audio file in current directory
        thinking_audio_path = "/Users/markbruckert/Documents/speech_to_text/thinking_audio.wav"
        if not os.path.exists(thinking_audio_path):
            # Try alternative names
            alt_paths = ["generated_thinking_audio.wav", "thinking.wav"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    thinking_audio_path = alt_path
                    break
            else:
                print("⚠️ Thinking audio file not found, skipping...")
                return None

        print("🤔 Playing thinking audio...")

        # Load and play thinking audio
        pygame.mixer.music.load(thinking_audio_path)
        pygame.mixer.music.play()

        return True

    except Exception as e:
        print(f"⚠️ Could not play thinking audio: {e}")
        return None


def stop_thinking_audio():
    """Stop thinking audio playback"""
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("🤫 Stopped thinking audio")
    except Exception as e:
        print(f"⚠️ Error stopping thinking audio: {e}")


def process_transcription_and_response(transcription):
    """Process the transcribed question, call API, and play response"""
    print(f"📄 Processing transcription: '{transcription}'")

    # Skip processing if transcription indicates an error or no speech
    if transcription in ["Audio too short", "No speech detected", "Error in transcription"]:
        print("⚠️ Skipping API call due to transcription issue.")
        return False

    # Start playing thinking audio while waiting for API response
    thinking_audio_playing = play_thinking_audio()

    try:
        # Call the character API endpoint with the transcribed text
        api_response = call_character_api(transcription)

        # Stop thinking audio once we get the response
        if thinking_audio_playing:
            stop_thinking_audio()

        if api_response and api_response.get('status') == 'success':
            print("✅ Received response from API.")

            # Print the text response if available
            if 'text_response' in api_response:
                print(f"💬 Response: '{api_response['text_response']}'")

            # Check for audio response
            if 'audio_base64' in api_response and api_response['audio_base64']:
                # Convert base64 to audio data and play
                audio_data = base64_to_audio_data(api_response['audio_base64'])
                if audio_data:
                    success = play_audio_response(audio_data)
                    if success:
                        print("🎵 Response played successfully.")
                        return True
                    else:
                        print("❌ Failed to play audio response.")
                        return False
                else:
                    print("❌ Failed to convert audio data.")
                    return False
            else:
                print("❌ No audio response received from API.")
                return False
        else:
            error_msg = api_response.get(
                'error', 'Unknown error') if api_response else 'No response'
            print(f"❌ API call failed: {error_msg}")
            return False

    except Exception as e:
        # Make sure to stop thinking audio if there's an error
        if thinking_audio_playing:
            stop_thinking_audio()
        print(f"❌ Error during API processing: {e}")
        return False


def transcribe_whisper(audio_path):
    try:
        print(f"🔍 Analyzing audio file: {audio_path}")

        # Check file size
        file_size = os.path.getsize(audio_path)
        print(f"📊 Audio file size: {file_size} bytes")

        if file_size < 1000:  # Less than 1KB is probably too short
            print("⚠️ Audio file too small, skipping transcription")
            return "Audio too short"

        # Load and analyze audio
        audio = whisper.load_audio(audio_path)
        print(
            f"📊 Audio length: {len(audio)} samples ({len(audio)/16000:.2f} seconds)")

        # Check if audio has sufficient volume
        audio_rms = np.sqrt(np.mean(audio**2))
        print(f"📊 Audio RMS: {audio_rms:.6f}")

        if audio_rms < 0.001:  # Very quiet audio
            print("⚠️ Audio appears to be very quiet")

        # Transcribe with more options
        result = model.transcribe(
            audio_path,
            language="en",
            verbose=True,
            word_timestamps=True,
            temperature=0.0  # More deterministic
        )

        transcript = result["text"].strip()
        print(f"🔍 Raw transcript: '{transcript}'")
        print(f"📊 Confidence segments: {len(result.get('segments', []))}")

        if not transcript:
            print("⚠️ Empty transcript - trying without language constraint")
            result = model.transcribe(
                audio_path, verbose=True, temperature=0.0)
            transcript = result["text"].strip()

        return transcript if transcript else "No speech detected"

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return "Error in transcription"


def callback(indata, frames, time_info, status):
    global is_recording, silence_start, recording, recording_start_time, is_processing_response

    # Don't start new recordings while processing API response
    if is_processing_response:
        return

    try:
        # Calculate multiple audio features for better voice detection
        volume_norm = np.linalg.norm(indata)
        volume_rms = np.sqrt(np.mean(indata**2))
        volume_max = np.max(np.abs(indata))

        # Use RMS for more stable voice detection
        current_volume = volume_rms

        # Debug: Print volume levels occasionally
        if int(time.time() * 10) % 100 == 0:  # Every 10 seconds
            print(
                f"🔊 Volume - RMS: {volume_rms:.4f}, Max: {volume_max:.4f}, Norm: {volume_norm:.4f} (threshold: {threshold})")

        # Voice activity detection with hysteresis
        voice_detected = current_volume > threshold

        # Additional check: ensure it's not just noise by checking consistency
        if voice_detected and len(recording) > 0:
            # Check if volume is consistently above a lower threshold
            recent_volumes = [np.sqrt(np.mean(chunk**2))
                              for chunk in recording[-5:]]  # Last 5 chunks
            if len(recent_volumes) >= 3:
                avg_recent_volume = np.mean(recent_volumes)
                if avg_recent_volume < threshold * 0.3:  # If recent average is too low, might be noise
                    voice_detected = False

        if voice_detected:
            if not is_recording:
                print(
                    f"🎙️ Voice detected (RMS: {current_volume:.4f}). Recording started.")
                is_recording = True
                recording_start_time = time.time()
            silence_start = None
            recording.append(indata.copy())
        elif is_recording:
            # Continue recording for a bit even during brief pauses
            recording.append(indata.copy())

            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration:
                recording_duration = time.time() - recording_start_time
                if recording_duration >= min_recording_duration:
                    print(
                        f"🛑 Silence detected. Recording stopped (duration: {recording_duration:.1f}s).")
                    is_recording = False
                    silence_start = None
                else:
                    print(
                        f"⚠️ Recording too short ({recording_duration:.1f}s), discarding...")
                    recording = []
                    is_recording = False
                    silence_start = None

    except Exception as e:
        print(f"❌ Callback error: {e}")


print("🎧 Listening... Speak to start recording.")
print(f"🔊 Volume threshold: {threshold}")
print("Press Ctrl+C to quit.")

start_session = time.time()

try:
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
        while not shutdown_flag:
            time.sleep(0.1)

            if not is_recording and recording and not is_transcribing:
                is_transcribing = True
                is_processing_response = True  # Prevent new recordings

                try:
                    print(f"📊 Processing {len(recording)} audio chunks...")

                    audio_data = np.concatenate(recording, axis=0)
                    print(f"📊 Total audio samples: {len(audio_data)}")

                    # Ensure audio is 1D and properly formatted
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()

                    # Normalize audio to prevent clipping
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.8

                    # Add padding
                    silence_padding = np.zeros(
                        sample_rate // 2, dtype=audio_data.dtype)  # 0.5 second padding
                    padded_audio = np.concatenate(
                        (silence_padding, audio_data, silence_padding))

                    # Create temporary file for audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        filename = temp_file.name

                    # Save as 16-bit PCM WAV
                    audio_int16 = (padded_audio * 32767).astype(np.int16)
                    write(filename, sample_rate, audio_int16)
                    print(f"✅ Saved as {filename}")

                    try:
                        # Transcribe using Whisper
                        print("📝 Transcribing...")
                        transcript = transcribe_whisper(filename)
                        print(f"📄 Transcript: '{transcript}'")

                        # Process transcription and play response - this blocks until audio finishes
                        audio_played = process_transcription_and_response(
                            transcript)

                        if audio_played:
                            print("🎵 Audio response completed.")
                    finally:
                        # Delete the temporary audio file
                        try:
                            os.unlink(filename)
                            print(f"🗑️ Deleted temporary file: {filename}")
                        except Exception as e:
                            print(f"⚠️ Could not delete file {filename}: {e}")

                    print()  # Add spacing before resuming listening

                except Exception as e:
                    print(f"⚠️ Processing error: {e}")
                    import traceback
                    traceback.print_exc()

                recording = []
                is_transcribing = False
                is_processing_response = False  # Allow new recordings
                print("🎧 Ready for next recording...")

            if time.time() - start_session > max_record_seconds:
                print("⏱️ Max session time reached. Exiting.")
                break

except Exception as e:
    print(f"❌ Main loop error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup pygame mixer
    try:
        pygame.mixer.quit()
    except:
        pass
    print("🏁 Program ended.")
