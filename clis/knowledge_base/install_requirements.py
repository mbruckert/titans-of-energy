import os
import subprocess

def install_requirements():
    print("Installing dependencies from requirements.txt...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    print("Downloading spaCy's English language model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

    print("Installation complete!")

if __name__ == "__main__":
    install_requirements()