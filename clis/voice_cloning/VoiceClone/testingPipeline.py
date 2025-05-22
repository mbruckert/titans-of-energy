import os
import shutil
from pathlib import Path
import tomllib
import csv
from datetime import datetime
import subprocess
from typing import Dict, List, Optional

class PipelineManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.results_dir = Path("Results")
        self.audio_preprocessing_dir = self.results_dir / "AudioPreProcessing"

        # Load configuration
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        # Initialize runtime tracking
        self.runtimes = {}

    def setup_directories(self):
        """Create required directories for each model."""
        # Get the base output directory from config
        base_output_dir = self.config.get("output_dir", "default_output")

        # Create the main results directory
        self.results_dir = Path("Results") / base_output_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Create the audio preprocessing directory
        self.audio_preprocessing_dir = self.results_dir / "AudioPreProcessing"
        os.makedirs(self.audio_preprocessing_dir, exist_ok=True)

        # Create model directories using just the model names
        for model in self.config["model"]:
            # Get just the model name (last part of the path)
            model_name = os.path.basename(model)

            # Create the model directory
            model_dir = self.results_dir / model_name
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory: {model_dir}")

    def parse_processing_order(self) -> List[str]:
        """Parse preprocessing order from config."""
        order = []
        if isinstance(self.config["preprocessing_order"], list):
            order = [step.lower().strip() for step in self.config["preprocessing_order"]]

        # Set defaults if empty
        if not order:
            if self.config.get("clean_audio", False):
                order.append("clean")
            if self.config.get("remove_silence", False):
                order.append("remove_silence")
            if self.config.get("enhance_audio", False):
                order.append("enhance")

        return order

    def preprocess_audio(self, input_audio: str) -> str:
        """Process audio according to specified order."""
        start_time = datetime.now()
        current_input = input_audio

        # Define script paths relative to the current directory
        script_dir = Path(__file__).parent
        audio_clean_path = script_dir / "cleanAudio.py"
        silence_removal_path = script_dir / "removeSilence.py"
        audio_enhance_path = script_dir / "enhanceAudio.py"

        for step in self.parse_processing_order():
            output_path = self.audio_preprocessing_dir / f"{step}_output.wav"

            if step == "clean":
                cmd = ["python", str(audio_clean_path), current_input, str(output_path)]
            elif step == "remove_silence":
                cmd = ["python", str(silence_removal_path), current_input, str(output_path)]
            elif step == "enhance":
                cmd = ["python", str(audio_enhance_path), current_input, str(output_path)]

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Completed {step} preprocessing")
                current_input = str(output_path)
            except subprocess.CalledProcessError as e:
                print(f"Error during {step} preprocessing:")
                print(f"Command: {' '.join(map(str, e.cmd))}")
                print(f"Return code: {e.returncode}")
                print(f"Error output: {e.stderr}")
                return None

        total_time = (datetime.now() - start_time).total_seconds()
        self.runtimes["audio_preprocessing"] = total_time
        return current_input

    def generate_model_configs(self, preprocessed_audio: str):
        """Generate TOML configs for each model."""
        for model in self.config["model"]:
            # Get just the model name (last part of the path)
            model_name = os.path.basename(model)

            # Create the model directory if it doesn't exist
            model_output_dir = self.results_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            output_dir = os.path.join(os.getcwd(), model_output_dir)


            model_config = {
                "model": model,
                "ref_audio": preprocessed_audio,
                "ref_text": self.config["ref_text"],
                "gen_text": self.config["gen_text"],
                "output_dir": str(output_dir)
            }

            model_toml_path = model_output_dir / "config.toml"
            with open(model_toml_path, "w", encoding="utf-8") as f:
                for key, value in model_config.items():
                    f.write(f'{key} = "{value}"\n')
            print(f"Generated config for {model}: {model_toml_path}")

    def run_models(self):

        for model in self.config["model"]:

            # Add Model Directory to the list for every model you have
            model_base_dir = {
                "F5TTS": "Models/F5-TTS",
                "tts_models/multilingual/multi-dataset/xtts_v2": "Models/TTS",
                "Zyphra/Zonos-v0.1-transformer": "Models/Zonos"
            }.get(model, "")

            if not model_base_dir:
                print(f"Warning: Unknown model base directory for {model}") #If you get this warning you need to add the model directory above
                continue

            # Get just the model name (last part of the path)
            original_cwd = os.getcwd()
            configPath = os.path.join(original_cwd, "Results", os.path.basename(model), "config.toml")
            print("Config: " + configPath)


            try:
                start_time = datetime.now()

                # Change to model directory and run
                os.chdir(model_base_dir)

                # Run the model with proper command
                cmd = f"python inference.py --config {configPath}"
                subprocess.run(cmd, shell=True, check=True)
                print("10. " + cmd)
                os.chdir(original_cwd)

                model_runtime = (datetime.now() - start_time).total_seconds()
                self.runtimes[model] = model_runtime
                print(f"Successfully ran {model}, runtime: {model_runtime:.2f}s")

            except subprocess.CalledProcessError as e:
                print(f"Error running model {model}: {e}")

    def run_similarity_test(self):
        """Run similarity test on model outputs."""
        reference_audio = self.config["ref_audio"]
        comparison_audios = [
            str(self.results_dir / model / "output.wav")
            for model in self.config["model"]
        ]

        # Run similarity test script
        cmd = f"python similarity_test.py {reference_audio} {' '.join(comparison_audios)}"
        print(f"\nRunning similarity test: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    def generate_similarity_report(self):
        """Generate similarity report CSV."""
        csv_path = self.results_dir / "similarity_report.csv"

        fieldnames = [
            "model",
            "runtime",
            "similarity_score",
            "clean_audio",
            "remove_silence",
            "enhance_audio"
        ]

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for model in self.config["model"]:
                row = {
                    "model": model,
                    "runtime": self.runtimes.get(model, 0) +
                             self.runtimes.get("audio_preprocessing", 0),
                    "similarity_score": None,  # Will be filled by similarity test
                    "clean_audio": self.config.get("clean_audio", False),
                    "remove_silence": self.config.get("remove_silence", False),
                    "enhance_audio": self.config.get("enhance_audio", False)
                }
                writer.writerow(row)

        print(f"Generated similarity report: {csv_path}")

def main():
    # Initialize and run the pipeline
    try:
        print("1. Loading Settings File .  .  .  . . . ...")
        pipeline = PipelineManager("config.toml")
        pipeline.setup_directories()
    except subprocess.CalledProcessError as e:
        print("v_v Settings File Failed to Load! v_v")
        print(e)
        return  # Exit early if settings fail
    print("^_^ Settings File Sucessfully Loaded ^_^")

    # Process audio
    try:
        print("2. Processing Reference Audio .  .  .  . . . ...")
        preprocessed_audio = pipeline.preprocess_audio(pipeline.config["ref_audio"])
        if preprocessed_audio is None:
            print("v_v Audio preprocessing failed! v_v")
            return  # Exit early if preprocessing fails
        print("^_^ Reference Audio Sucessfully Processed ^_^")
        print(f"Final preprocessed audio: {preprocessed_audio}")
    except subprocess.CalledProcessError as e:
        print("v_v Settings File Failed to Load! v_v")
        print(e)
        return  # Exit early if preprocessing fails

    # Generate model configs
    try:
        print("3. Generating Model Configs .  .  .  . . . ...")
        pipeline.generate_model_configs(preprocessed_audio)
    except subprocess.CalledProcessError as e:
        print("v_v Model Configs Failed to Generate! v_v")
        print(e)
    print("^_^ Model Configs Sucessfully Generated ^_^")
    """
    # Run models
    try:
        print("4. Running Individual Models .  .  .  . . . ...")
        pipeline.run_models()
    except subprocess.CalledProcessError as e:
        print("v_v Model Failed to Run! v_v")
        print(e)

    # Run similarity test
    try:
        print("5. Running Similarity Test .  .  .  . . . ...")
        pipeline.run_similarity_test()
    except subprocess.CalledProcessError as e:
        print("v_v Similarity Test Failed to Run! v_v")
        print(e)
    print("^_^ Similarity Test Completed Sucessfully ^_^")

    # Generate final report
    try:
        print("6. Generating Final Report .  .  .  . . . ...")
        pipeline.generate_similarity_report()
    except subprocess.CalledProcessError as e:
        print("v_v Final Report Failed to Generate! v_v")
        print(e)
    print("^_^ Final Report Completed Sucessfully ^_^")
    """
if __name__ == "__main__":
    main()
