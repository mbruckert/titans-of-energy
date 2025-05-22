import os
from pathlib import Path
from typing import Optional
import torch
import subprocess
import time
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def main():
    # Set up environment variables
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",  # Adjust GPU index as needed
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"
    }

    # Define paths and parameters
    reference_audio = "/home/styx/Videos/VoiceClone/Models/Zonos/Tests/Test5.6Input.wav"  # Your reference audio file
    output_path = "/home/styx/Videos/VoiceClone/Models/Zonos/Tests/Test5.6Output.wav"

    # Load Zonos model
    try:
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    except Exception as e:
        print(f"Error loading Zonos model: {str(e)}")
        return

    # Load reference audio and create speaker embedding
    try:
        wav, sampling_rate = torchaudio.load(reference_audio)
        speaker = model.make_speaker_embedding(wav, sampling_rate)
    except Exception as e:
        print(f"Error loading reference audio: {str(e)}")
        return
    start_time = time.perf_counter()
    # Generate speech using Zonos
    try:
        cond_dict = make_cond_dict(
            text="My name is Robert J Oppenheimer and welcome to the Titans of Energy statue exhibit at the Oak Ridge National Laboratory.",
            speaker=speaker,
            language="en-us"
        )
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

        # Verify output path is a file
        if not output_path.endswith('.wav'):
            output_path += '.wav'

        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        print(f"Voice cloning successful! Output saved to: {output_path}")

    except RuntimeError as e:
        if "espeak not installed" in str(e):
            print("\nError: espeak-ng is required for Zonos TTS.")
            print("Please install it using one of these commands:")
            print("Ubuntu/Debian: sudo apt install espeak-ng")
            print("MacOS: brew install espeak-ng")
        else:
            print(f"\nError during speech generation: {str(e)}")

    runtime = time.perf_counter() - start_time
    print(f"Script completed in {runtime:.3f} seconds")

if __name__ == "__main__":
    main()
