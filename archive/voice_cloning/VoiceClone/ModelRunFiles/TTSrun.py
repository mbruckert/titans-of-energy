import os
from pathlib import Path
from typing import Optional
from torch.serialization import add_safe_globals, safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
import torch
import subprocess
import time

def main():
    torch.serialization.add_safe_globals([XttsConfig])

    env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1",  # Adjust GPU index as needed
            "COQUI_TOS_AGREED": "1",
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': "1"
        }

    # Run inference command
    cmd = [
            "tts",
            "--text", "My name is Robert J Oppenheimer and welcome to the Titans of Energy statue exhibit at the Oak Ridge National Laboratory.",
            "--reference_wav", "/home/styx/Videos/VoiceClone/Models/TTS/Tests/Test5.6Input.wav",
            "--speaker_wav", "/home/styx/Videos/VoiceClone/Models/TTS/Tests/Test5.6Input.wav",
            "--out_path", "/home/styx/Videos/VoiceClone/Models/TTS/Tests/Test5.6Output.wav",
            "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
            "--language_idx", "en"
        ]
    start_time = time.perf_counter()
    print(f"\nRunning command: {cmd}")
    try:
        # Run the inference
        result = subprocess.run(
            cmd,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )

        print(f"TTS generation successful!")
        print(f"Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"Error running XTTS-v2:")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")

    runtime = time.perf_counter() - start_time
    print(f"Script completed in {runtime} seconds")

if __name__ == "__main__":
    main()
