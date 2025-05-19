import sys
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

def calculate_similarities(reference_path, comparison_paths):
    """
    Calculate similarities between a reference audio file and multiple comparison files.

    Args:
        reference_path (str): Path to the reference audio file
        comparison_paths (list[str]): List of paths to comparison audio files

    Returns:
        dict: Dictionary mapping comparison file names to their similarity scores
    """
    # Initialize the voice encoder
    encoder = VoiceEncoder()

    # Process the reference audio
    ref_wav = preprocess_wav(reference_path)
    ref_embedding = encoder.embed_utterance(ref_wav)

    # Dictionary to store similarities
    similarities = {}

    # Process each comparison audio file
    for comp_path in comparison_paths:
        try:
            # Load and process comparison audio
            comp_wav = preprocess_wav(comp_path)

            # Calculate similarity score (dot product between embeddings)
            comp_embedding = encoder.embed_utterance(comp_wav)
            similarity = np.dot(ref_embedding, comp_embedding) / (
                np.linalg.norm(ref_embedding) *
                np.linalg.norm(comp_embedding)
            )

            # Store the result
            similarities[Path(comp_path).name] = similarity

        except Exception as e:
            print(f"Error processing {comp_path}: {str(e)}")
            similarities[Path(comp_path).name] = None

    return similarities

def main():
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python similarity_test.py <reference_audio> <comparison_audio1> [comparison_audio2 ...]")
        sys.exit(1)

    reference_audio = sys.argv[1]
    comparison_audios = sys.argv[2:]

    # Calculate similarities
    results = calculate_similarities(reference_audio, comparison_audios)

    # Print results
    print("\nSimilarity Results:")
    print("-" * 50)
    for file_name, similarity in results.items():
        if similarity is not None:
            print(f"{file_name}: {similarity:.4f}")
        else:
            print(f"{file_name}: Error processing file")

if __name__ == "__main__":
    main()
