import os
from pathlib import Path
from typing import Optional

def setup_environment():
    """Set up required environment variables"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust based on your GPU availability

def create_config_file(config_path: Path, model: str, ref_audio: str, ref_text: str, gen_text: str):
    """Create configuration file for F5-TTS inference"""
    config_content = f"""[inference]
model_name = "{model}"
reference_audio = "{ref_audio}"
reference_text = "{ref_text}"
generate_text = "{gen_text}"

[output]
output_dir = "{config_path.parent}"
"""

    with open(config_path, "w") as f:
        f.write(config_content)

def main():
    # Set up environment
    setup_environment()

    # Configuration values
    model = "F5TTS"
    ref_audio = ""
    ref_text = "of course the initial discovery and its interpretation in early 1939 attracted everybody's interest."
    gen_text = "My name is Robert J Oppenheimer and welcome to the Titans of Energy statue exhibit at the Oak Ridge National Laboratory."
    output_dir = "outputs"

    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Create configuration file
    config_path = output_dir_path / "inference.toml"
    create_config_file(
        config_path,
        model=model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text
    )

    # Run inference command
    cmd = f"f5-tts_infer-cli -c {config_path}"
    print(f"\nRunning command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
