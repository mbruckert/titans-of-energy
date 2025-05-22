# audio_clean.py
import sys
import numpy as np
import resampy
import soundfile
import torch
from df.enhance import enhance, init_df

def main():
    if len(sys.argv) != 3:
        print("Usage: python audio_clean.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Initialize the model
    model, state, _ = init_df()

    # Read and process audio
    inputFile, sr = soundfile.read(input_file, always_2d=True)

    # Resample if necessary
    inputFile = inputFile if sr == state.sr() else resampy.resample(inputFile, sr, state.sr())
    sr = state.sr()

    # Convert to float32 and create tensor
    inputFile = inputFile.astype(np.float32).T
    inputFile = torch.from_numpy(inputFile)

    # Enhance audio
    outputFile = enhance(model, state, inputFile)

    # Convert back to numpy array
    outputFile = outputFile.detach().cpu().numpy()
    outputFile = np.squeeze(outputFile.T)

    # Save output
    soundfile.write(output_file, outputFile, sr)

if __name__ == "__main__":
    main()
