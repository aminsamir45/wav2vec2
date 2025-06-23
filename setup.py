from setuptools import setup, find_packages

setup(
    name="efficient-wav2vec2",
    version="0.1.0",
    description="Efficient wav2vec 2.0 for Ultra-Low Resource Speech Recognition",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "fairseq>=0.12.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
        "omegaconf>=2.1.0",
        "hydra-core>=1.1.0",
        "jiwer>=2.3.0",  # For WER calculation
        "phonemizer>=3.0.0",  # For phoneme-aware masking
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
) 