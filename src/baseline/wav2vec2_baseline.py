"""
Baseline wav2vec 2.0 implementation for speech recognition.
This serves as our starting point before implementing improvements.
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import jiwer
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Wav2Vec2Baseline:
    """Baseline wav2vec 2.0 implementation using HuggingFace transformers"""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device="auto"):
        """
        Initialize the baseline model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model: {model_name}")
    
    def preprocess_audio(self, audio_path_or_array, sampling_rate=16000):
        """
        Preprocess audio for wav2vec 2.0
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            sampling_rate: Target sampling rate
            
        Returns:
            Preprocessed audio tensor
        """
        if isinstance(audio_path_or_array, (str, Path)):
            # Load audio file
            waveform, sr = torchaudio.load(audio_path_or_array)
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                waveform = resampler(waveform)
        else:
            # Assume numpy array
            waveform = torch.tensor(audio_path_or_array, dtype=torch.float32)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze()
    
    def transcribe(self, audio_input, sampling_rate=16000):
        """
        Transcribe audio to text
        
        Args:
            audio_input: Audio file path or numpy array
            sampling_rate: Audio sampling rate
            
        Returns:
            Transcription string
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_input, sampling_rate)
        
        # Process with wav2vec 2.0 processor
        inputs = self.processor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.lower().strip()
    
    def evaluate_on_dataset(self, dataset, max_samples=None):
        """
        Evaluate the model on a dataset
        
        Args:
            dataset: HuggingFace dataset with 'audio' and 'text' columns
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        predictions = []
        references = []
        
        logger.info(f"Evaluating on {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(dataset)}")
            
            # Get audio and reference text
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            reference = sample["text"].lower().strip()
            
            # Transcribe
            try:
                prediction = self.transcribe(audio, sampling_rate)
                predictions.append(prediction)
                references.append(reference)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate WER
        wer = jiwer.wer(references, predictions)
        
        # Calculate CER
        cer = jiwer.cer(references, predictions)
        
        results = {
            "wer": wer,
            "cer": cer,
            "num_samples": len(predictions),
            "predictions": predictions[:10],  # First 10 for inspection
            "references": references[:10],
        }
        
        logger.info(f"WER: {wer:.4f}, CER: {cer:.4f}")
        
        return results

def create_low_resource_subset(dataset, duration_minutes):
    """
    Create a low-resource subset of the dataset
    
    Args:
        dataset: Full dataset
        duration_minutes: Target duration in minutes
        
    Returns:
        Subset of the dataset
    """
    target_duration = duration_minutes * 60  # Convert to seconds
    current_duration = 0
    selected_indices = []
    
    for i, sample in enumerate(dataset):
        audio_duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        if current_duration + audio_duration <= target_duration:
            selected_indices.append(i)
            current_duration += audio_duration
        else:
            break
    
    logger.info(f"Selected {len(selected_indices)} samples for {duration_minutes}-minute subset")
    logger.info(f"Total duration: {current_duration/60:.2f} minutes")
    
    return dataset.select(selected_indices)

def main():
    """Example usage and baseline evaluation"""
    
    # Initialize baseline model
    baseline = Wav2Vec2Baseline()
    
    # Load LibriSpeech test set
    logger.info("Loading LibriSpeech test set...")
    test_dataset = load_dataset("librispeech_asr", "clean", split="test")
    
    # Evaluate on a small subset first
    small_test = test_dataset.select(range(100))
    results = baseline.evaluate_on_dataset(small_test)
    
    print("\nBaseline Results:")
    print(f"WER: {results['wer']:.4f}")
    print(f"CER: {results['cer']:.4f}")
    print(f"Samples evaluated: {results['num_samples']}")
    
    print("\nExample predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"Reference: {results['references'][i]}")
        print(f"Prediction: {results['predictions'][i]}")
        print()

if __name__ == "__main__":
    main() 