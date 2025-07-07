import argparse
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import keyboard
from scipy.io.wavfile import write
from transformers import (Wav2Vec2Processor,Wav2Vec2ForCTC,HubertForCTC)

# ---------------- CLI Arguments ----------------
parser = argparse.ArgumentParser(description="🎤 Voice-activated recorder and transcriber using Wav2Vec2 or HuBERT")

parser.add_argument("--model", choices=["wav2vec2", "hubert"], default="hubert", help="Choose STT model")
parser.add_argument("--threshold", type=float, default=0.3, help="Volume threshold to start recording")
parser.add_argument("--silence", type=float, default=2.0, help="Seconds of silence before stopping")
parser.add_argument("--max_time", type=int, default=300, help="Max session time in seconds")
parser.add_argument("--output", type=str, default=".", help="Output folder for recordings")

args = parser.parse_args()

# ---------------- Model Mapping ----------------
model_map = {
    "wav2vec2": "facebook/wav2vec2-large-960h",
    "hubert": "facebook/hubert-large-ls960-ft"
}
hf_model_id = model_map[args.model]

# ---------------- Configuration ----------------
sample_rate = 16000
threshold = args.threshold
silence_duration = args.silence
max_record_seconds = args.max_time
output_folder = args.output

recording = []
is_recording = False
is_transcribing = False
silence_start = None

# ---------------- Load Model ----------------
print(f"📦 Loading model: {args.model}...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-960h" if args.model == "wav2vec2" else "facebook/hubert-large-ls960-ft"
)
model = (
    Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    if args.model == "wav2vec2"
    else HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
)
# ---------------- Helper Functions ----------------
def get_next_filename(prefix="recording", ext=".wav", folder="."):
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        filename = os.path.join(folder, f"{prefix}{i}{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

def transcribe_audio(audio_path):
    speech, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz sample rate, got {sr}")
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    
    transcription = processor.decode(predicted_ids[0])
    return transcription

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

# ---------------- Main Loop ----------------
print("🎧 Listening... Speak to start recording.")
print("Press 'q' to quit.")

start_session = time.time()

with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
    try:
        while True:
            time.sleep(0.1)

            if not is_recording and recording and not is_transcribing:
                is_transcribing = True

                audio_data = np.concatenate(recording, axis=0)
                silence_padding = np.zeros((sample_rate, 1), dtype=audio_data.dtype)
                padded_audio = np.concatenate((silence_padding, audio_data, silence_padding), axis=0)

                filename = get_next_filename(folder=output_folder)
                write(filename, sample_rate, padded_audio)
                print(f"✅ Saved as {filename}")

                print("📝 Transcribing...")
                try:
                    transcript = transcribe_audio(filename)
                    print(f"📄 Transcript: {transcript}")
                except Exception as e:
                    print(f"⚠️ Transcription error: {e}")

                recording = []
                is_transcribing = False

            if keyboard.is_pressed('q'):
                print("🛑 'q' pressed. Exiting program.")
                break

            if time.time() - start_session > max_record_seconds:
                print("⏱️ Max session time reached. Exiting.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
