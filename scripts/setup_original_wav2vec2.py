#!/usr/bin/env python3
"""
Script to download and set up the original wav2vec 2.0 model and code.
This will help us establish the baseline before implementing improvements.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

def download_model(model_name, save_dir):
    """Download pre-trained wav2vec 2.0 models"""
    models = {
        "wav2vec2_base": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
        "wav2vec2_large": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
        "wav2vec2_base_960h": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_base_960h.pt",
    }
    
    if model_name not in models:
        print(f"Available models: {list(models.keys())}")
        return False
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"{model_name}.pt"
    
    if model_path.exists():
        print(f"Model {model_name} already exists at {model_path}")
        return True
    
    print(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(models[model_name], model_path)
        print(f"Successfully downloaded {model_name} to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False

def setup_fairseq():
    """Install fairseq if not already installed"""
    try:
        import fairseq
        print("fairseq already installed")
        return True
    except ImportError:
        print("Installing fairseq...")
        try:
            subprocess.run(["pip", "install", "fairseq"], check=True)
            print("fairseq installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing fairseq: {e}")
            return False

def download_librispeech_subsets():
    """Download LibriSpeech low-resource subsets for experiments"""
    from datasets import load_dataset
    
    print("Downloading LibriSpeech datasets...")
    
    # Download the datasets we'll use for experiments
    datasets_to_download = [
        "librispeech_asr",  # We'll create subsets from this
    ]
    
    for dataset_name in datasets_to_download:
        try:
            print(f"Loading {dataset_name}...")
            dataset = load_dataset(dataset_name, "clean", split="train.100")  # 100-hour subset
            print(f"Successfully loaded {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")

def main():
    """Main setup function"""
    print("Setting up original wav2vec 2.0 baseline...")
    
    # Create directories
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup fairseq
    if not setup_fairseq():
        print("Failed to setup fairseq")
        return
    
    # Download pre-trained models
    models_to_download = ["wav2vec2_base", "wav2vec2_base_960h"]
    
    for model in models_to_download:
        download_model(model, models_dir)
    
    # Download datasets
    download_librispeech_subsets()
    
    print("\nSetup complete!")
    print("Next steps:")
    print("1. Run baseline experiments with: python scripts/run_baseline.py")
    print("2. Start implementing improvements in src/improvements/")

if __name__ == "__main__":
    main() 