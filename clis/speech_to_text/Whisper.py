import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import keyboard  # pip install keyboard
import whisper  # pip install openai-whisper

# Configuration
sample_rate = 16000
threshold = 0.3
silence_duration = 2
max_record_seconds = 300

recording = []
is_recording = False
silence_start = None

# Load Whisper model
print("📦 Loading Whisper-large model...")
model = whisper.load_model("large")  # You can also try "large-v2"

def get_next_filename(prefix="recording", ext=".wav", folder="."):
    i = 1
    while True:
        filename = os.path.join(folder, f"{prefix}{i}{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

def transcribe_whisper(audio_path):
    result = model.transcribe(audio_path, language="en")  # force English for faster decoding
    return result["text"]

def callback(indata, frames, time_info, status):
    global is_recording, silence_start, recording

    volume_norm = np.linalg.norm(indata)
    if volume_norm > threshold:
        if not is_recording:
            print("🎙️ Voice detected. Recording started.")
            is_recording = True
        silence_start = None
        recording.append(indata.copy())
    elif is_recording:
        if silence_start is None:
            silence_start = time.time()
        elif time.time() - silence_start > silence_duration:
            print("🛑 Silence detected. Recording stopped.")
            is_recording = False
            silence_start = None

print("🎧 Listening... Speak to start recording.")
print("Press 'q' to quit.")

start_session = time.time()

with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
    try:
        while True:
            time.sleep(0.1)
            
            if not is_recording and recording:
                audio_data = np.concatenate(recording, axis=0)
                silence_padding = np.zeros((sample_rate, 1), dtype=audio_data.dtype)
                padded_audio = np.concatenate((silence_padding, audio_data, silence_padding), axis=0)

                filename = get_next_filename()
                write(filename, sample_rate, padded_audio)
                print(f"✅ Saved as {filename}")

                # Transcribe using Whisper
                print("📝 Transcribing...")
                try:
                    transcript = transcribe_whisper(filename)
                    print(f"📄 Transcript: {transcript}")
                except Exception as e:
                    print(f"⚠️ Transcription error: {e}")

                recording = []

            if keyboard.is_pressed('q'):
                print("🛑 'q' pressed. Exiting program.")
                break

            if time.time() - start_session > max_record_seconds:
                print("⏱️ Max session time reached. Exiting.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
