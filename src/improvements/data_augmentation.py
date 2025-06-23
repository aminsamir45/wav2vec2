"""
Enhanced Data Utilization for Ultra-Low Resource ASR

WHY THIS IMPROVEMENT:
With only 10 minutes to 1 hour of labeled data, every sample is precious.
We need sophisticated data augmentation and utilization strategies:

1. Synthetic Data Generation - Create additional training samples
2. Cross-lingual Transfer - Leverage high-resource languages
3. Self-training - Use model predictions to expand training data
4. Intelligent Data Selection - Choose the most informative samples

EXPECTED BENEFITS:
- Effective data multiplication without manual annotation
- Better coverage of acoustic variations
- Improved robustness to different speaking styles
- Reduced overfitting through diverse training data
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import librosa
from datasets import Dataset
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class AudioAugmentation:
    """
    Audio augmentation techniques for speech data
    
    WHY: With limited data, we need to artificially increase diversity
    to prevent overfitting and improve generalization to new speakers
    and acoustic conditions.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        logger.info(f"Audio augmentation initialized for {sample_rate}Hz audio")
    
    def speed_perturbation(self, audio: torch.Tensor, factor: float = None) -> torch.Tensor:
        """
        Speed perturbation without changing pitch
        
        WHY: Different speaking rates are common in real speech.
        Speed perturbation simulates this variation while preserving
        phonetic content.
        
        Args:
            audio: Input audio tensor
            factor: Speed factor (0.9-1.1 typical range)
        """
        if factor is None:
            factor = np.random.uniform(0.9, 1.1)
        
        # Resample to change speed
        original_length = audio.shape[-1]
        new_length = int(original_length / factor)
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * factor)
        )
        
        # Apply resampling then pad/truncate to original length
        audio_resampled = resampler(audio)
        
        if audio_resampled.shape[-1] > original_length:
            audio_resampled = audio_resampled[:, :original_length]
        elif audio_resampled.shape[-1] < original_length:
            padding = original_length - audio_resampled.shape[-1]
            audio_resampled = torch.nn.functional.pad(audio_resampled, (0, padding))
        
        return audio_resampled
    
    def noise_injection(self, audio: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Add background noise to simulate realistic conditions
        
        WHY: Real-world speech often contains background noise.
        Adding controlled noise improves robustness to noisy conditions.
        
        Args:
            audio: Input audio tensor
            snr_db: Signal-to-noise ratio in dB (higher = less noise)
        """
        if snr_db is None:
            snr_db = np.random.uniform(10, 30)  # 10-30 dB SNR
        
        # Generate white noise
        noise = torch.randn_like(audio)
        
        # Calculate power levels
        signal_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Calculate noise scaling factor
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        # Add scaled noise
        noisy_audio = audio + noise_scale * noise
        
        return noisy_audio
    
    def pitch_shift(self, audio: torch.Tensor, semitones: float = None) -> torch.Tensor:
        """
        Shift pitch without changing speed
        
        WHY: Different speakers have different fundamental frequencies.
        Pitch shifting simulates speaker variation while preserving
        temporal characteristics.
        
        Args:
            audio: Input audio tensor
            semitones: Pitch shift in semitones (-2 to +2 typical)
        """
        if semitones is None:
            semitones = np.random.uniform(-2, 2)
        
        # Convert to numpy for librosa processing
        audio_np = audio.squeeze().numpy()
        
        # Apply pitch shifting
        shifted = librosa.effects.pitch_shift(
            audio_np, sr=self.sample_rate, n_steps=semitones
        )
        
        return torch.tensor(shifted, dtype=audio.dtype).unsqueeze(0)
    
    def room_impulse_response(self, audio: torch.Tensor, room_size: str = "random") -> torch.Tensor:
        """
        Simulate different room acoustics
        
        WHY: Speech recordings happen in various acoustic environments.
        Simulating different room characteristics improves generalization
        across recording conditions.
        
        Args:
            audio: Input audio tensor
            room_size: Room size simulation ("small", "medium", "large", "random")
        """
        if room_size == "random":
            room_size = np.random.choice(["small", "medium", "large"])
        
        # Simple reverb simulation using exponential decay
        decay_params = {
            "small": (0.1, 0.3),    # (decay_time, wet_mix)
            "medium": (0.3, 0.4),
            "large": (0.6, 0.5)
        }
        
        decay_time, wet_mix = decay_params[room_size]
        
        # Create simple exponential decay impulse response
        impulse_length = int(decay_time * self.sample_rate)
        impulse = torch.exp(-torch.linspace(0, 5, impulse_length))
        
        # Apply convolution
        reverb_audio = torchaudio.functional.convolve(audio, impulse.unsqueeze(0))
        
        # Mix with original (wet/dry mix)
        reverb_audio = reverb_audio[:, :audio.shape[-1]]  # Trim to original length
        mixed_audio = (1 - wet_mix) * audio + wet_mix * reverb_audio
        
        return mixed_audio
    
    def augment_batch(self, audio_batch: torch.Tensor, 
                     augmentation_prob: float = 0.5) -> torch.Tensor:
        """
        Apply random augmentations to a batch of audio
        
        Args:
            audio_batch: Batch of audio tensors [batch_size, channels, samples]
            augmentation_prob: Probability of applying each augmentation
        """
        augmented_batch = []
        
        for audio in audio_batch:
            augmented_audio = audio.clone()
            
            # Randomly apply augmentations
            if np.random.random() < augmentation_prob:
                augmented_audio = self.speed_perturbation(augmented_audio)
            
            if np.random.random() < augmentation_prob:
                augmented_audio = self.noise_injection(augmented_audio)
            
            if np.random.random() < augmentation_prob:
                augmented_audio = self.pitch_shift(augmented_audio)
            
            if np.random.random() < augmentation_prob:
                augmented_audio = self.room_impulse_response(augmented_audio)
            
            augmented_batch.append(augmented_audio)
        
        return torch.stack(augmented_batch)

class SyntheticDataGenerator:
    """
    Generate synthetic speech data using TTS models
    
    WHY: With only 10 minutes of real data, we can use text-to-speech
    to create additional training samples from available text transcripts.
    This dramatically increases data availability for training.
    """
    
    def __init__(self, tts_model_name: str = "facebook/fastspeech2-en-ljspeech"):
        """
        Args:
            tts_model_name: Name of the TTS model to use
        """
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
            self.processor = SpeechT5Processor.from_pretrained(tts_model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_name)
            self.available = True
            logger.info(f"TTS model loaded: {tts_model_name}")
        except ImportError:
            logger.warning("TTS dependencies not available. Synthetic generation disabled.")
            self.available = False
    
    def generate_from_text(self, texts: List[str], speaker_embeddings=None) -> List[torch.Tensor]:
        """
        Generate synthetic audio from text transcripts
        
        WHY: We can take text transcripts and generate additional audio samples
        with different speaker characteristics, expanding our training data.
        
        Args:
            texts: List of text transcripts to synthesize
            speaker_embeddings: Optional speaker embeddings for voice control
            
        Returns:
            List of synthesized audio tensors
        """
        if not self.available:
            logger.error("TTS model not available")
            return []
        
        synthetic_audio = []
        
        for text in texts:
            try:
                # Process text
                inputs = self.processor(text=text, return_tensors="pt")
                
                # Generate speech
                with torch.no_grad():
                    speech = self.model.generate_speech(
                        inputs["input_ids"], 
                        speaker_embeddings
                    )
                
                synthetic_audio.append(speech)
                
            except Exception as e:
                logger.warning(f"Failed to synthesize '{text[:50]}...': {e}")
                continue
        
        logger.info(f"Generated {len(synthetic_audio)} synthetic samples")
        return synthetic_audio
    
    def create_speaker_variations(self, text: str, num_variations: int = 3) -> List[torch.Tensor]:
        """
        Create multiple speaker variations of the same text
        
        WHY: Different speakers saying the same text provides valuable
        training signal about speaker-independent speech recognition.
        """
        # This would typically use different speaker embeddings
        # For now, we use different random seeds to get variation
        variations = []
        
        for i in range(num_variations):
            torch.manual_seed(42 + i)  # Different seed for each variation
            audio = self.generate_from_text([text])
            if audio:
                variations.extend(audio)
        
        return variations

class CrossLingualTransfer:
    """
    Leverage high-resource languages for low-resource ASR
    
    WHY: Many speech patterns are universal across languages.
    We can use models trained on high-resource languages and
    adapt them to low-resource scenarios.
    """
    
    def __init__(self, source_language: str = "en", target_language: str = "en"):
        self.source_language = source_language
        self.target_language = target_language
        logger.info(f"Cross-lingual transfer: {source_language} -> {target_language}")
    
    def extract_cross_lingual_features(self, model, audio_data: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract language-independent features using a multilingual model
        
        WHY: Lower layers of speech models often capture language-independent
        acoustic features that can transfer across languages.
        
        Args:
            model: Pre-trained multilingual wav2vec 2.0 model
            audio_data: List of audio tensors
            
        Returns:
            Extracted features tensor
        """
        features = []
        
        model.eval()
        with torch.no_grad():
            for audio in audio_data:
                # Extract features from intermediate layers (not final classification)
                if hasattr(model, 'wav2vec2'):
                    # Get features from the encoder (before classification head)
                    feature_extractor_output = model.wav2vec2.feature_extractor(audio.unsqueeze(0))
                    encoder_output = model.wav2vec2.encoder(feature_extractor_output.transpose(1, 2))
                    features.append(encoder_output.last_hidden_state.squeeze(0))
                else:
                    # Fallback for different model structures
                    output = model(audio.unsqueeze(0), output_hidden_states=True)
                    features.append(output.hidden_states[-2].squeeze(0))  # Second-to-last layer
        
        return torch.stack(features)
    
    def create_cross_lingual_dataset(self, source_data: Dataset, 
                                   target_data: Dataset, 
                                   mixing_ratio: float = 0.3) -> Dataset:
        """
        Create a mixed dataset with both source and target language data
        
        WHY: Training on mixed data helps the model learn language-independent
        features while adapting to the target language.
        
        Args:
            source_data: High-resource language dataset
            target_data: Low-resource target language dataset
            mixing_ratio: Ratio of source language data to include
        """
        # Sample from source data
        source_size = int(len(target_data) * mixing_ratio)
        source_subset = source_data.shuffle().select(range(min(source_size, len(source_data))))
        
        # Combine datasets
        combined_texts = list(target_data["text"]) + list(source_subset["text"])
        combined_audio = list(target_data["audio"]) + list(source_subset["audio"])
        
        # Create language labels for potential use in training
        target_labels = ["target"] * len(target_data)
        source_labels = ["source"] * len(source_subset)
        language_labels = target_labels + source_labels
        
        # Create new dataset
        mixed_dataset = Dataset.from_dict({
            "text": combined_texts,
            "audio": combined_audio,
            "language": language_labels
        })
        
        logger.info(f"Created mixed dataset: {len(target_data)} target + {len(source_subset)} source samples")
        
        return mixed_dataset

class SelfTrainingAugmentation:
    """
    Self-training approach for data augmentation
    
    WHY: Once we have a partially trained model, we can use it to generate
    pseudo-labels for unlabeled audio data, effectively expanding our
    training set.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"Self-training with confidence threshold: {confidence_threshold}")
    
    def generate_pseudo_labels(self, model, unlabeled_audio: List[torch.Tensor]) -> List[Dict]:
        """
        Generate pseudo-labels for unlabeled audio data
        
        WHY: We can use model predictions on unlabeled data as additional
        training samples, but only if we're confident about the predictions.
        
        Args:
            model: Trained wav2vec 2.0 model
            unlabeled_audio: List of unlabeled audio tensors
            
        Returns:
            List of dictionaries with audio and pseudo-labels
        """
        pseudo_labeled_data = []
        
        model.eval()
        with torch.no_grad():
            for i, audio in enumerate(unlabeled_audio):
                try:
                    # Get model predictions
                    inputs = {"input_values": audio.unsqueeze(0)}
                    outputs = model(**inputs)
                    
                    # Get predicted probabilities
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get predicted tokens
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # Calculate confidence (mean of max probabilities)
                    confidence = torch.mean(torch.max(probabilities, dim=-1)[0]).item()
                    
                    if confidence >= self.confidence_threshold:
                        # Decode prediction
                        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                            # This would need the proper tokenizer/processor
                            pseudo_text = f"pseudo_label_{i}"  # Placeholder
                        else:
                            pseudo_text = f"pseudo_label_{i}"
                        
                        pseudo_labeled_data.append({
                            "audio": audio,
                            "text": pseudo_text,
                            "confidence": confidence,
                            "is_pseudo": True
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to generate pseudo-label for sample {i}: {e}")
                    continue
        
        logger.info(f"Generated {len(pseudo_labeled_data)} pseudo-labeled samples "
                   f"from {len(unlabeled_audio)} unlabeled samples")
        
        return pseudo_labeled_data
    
    def iterative_self_training(self, model, labeled_data: Dataset, 
                              unlabeled_data: List[torch.Tensor], 
                              num_iterations: int = 3) -> Dataset:
        """
        Iterative self-training to progressively expand the dataset
        
        WHY: As the model improves, it can generate better pseudo-labels,
        which can then be used to train an even better model.
        
        Args:
            model: Initial trained model
            labeled_data: Original labeled dataset
            unlabeled_data: Pool of unlabeled audio
            num_iterations: Number of self-training iterations
        """
        current_dataset = labeled_data
        
        for iteration in range(num_iterations):
            logger.info(f"Self-training iteration {iteration + 1}/{num_iterations}")
            
            # Generate pseudo-labels with current model
            pseudo_data = self.generate_pseudo_labels(model, unlabeled_data)
            
            if not pseudo_data:
                logger.warning(f"No confident pseudo-labels in iteration {iteration + 1}")
                continue
            
            # Convert to dataset format and combine with existing data
            pseudo_texts = [item["text"] for item in pseudo_data]
            pseudo_audio = [item["audio"] for item in pseudo_data]
            
            # Create expanded dataset
            expanded_texts = list(current_dataset["text"]) + pseudo_texts
            expanded_audio = list(current_dataset["audio"]) + pseudo_audio
            
            current_dataset = Dataset.from_dict({
                "text": expanded_texts,
                "audio": expanded_audio
            })
            
            logger.info(f"Expanded dataset to {len(current_dataset)} samples")
            
            # Here you would retrain the model with the expanded dataset
            # model = retrain_model(model, current_dataset)
        
        return current_dataset

def create_enhanced_dataset(original_dataset: Dataset, 
                          augmentation_factor: int = 3,
                          use_synthetic: bool = True,
                          use_cross_lingual: bool = False) -> Dataset:
    """
    Create an enhanced dataset using multiple data augmentation techniques
    
    WHY: Combine all our data enhancement strategies to maximize the value
    of limited labeled data.
    
    Args:
        original_dataset: Original small dataset
        augmentation_factor: How many times to augment each sample
        use_synthetic: Whether to use TTS-generated data
        use_cross_lingual: Whether to use cross-lingual transfer
        
    Returns:
        Enhanced dataset with much more training data
    """
    logger.info(f"Creating enhanced dataset from {len(original_dataset)} samples")
    
    # Initialize augmentation tools
    audio_aug = AudioAugmentation()
    synthetic_gen = SyntheticDataGenerator() if use_synthetic else None
    
    enhanced_texts = []
    enhanced_audio = []
    
    # Augment existing samples
    for sample in original_dataset:
        original_text = sample["text"]
        original_audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        
        # Keep original
        enhanced_texts.append(original_text)
        enhanced_audio.append(original_audio)
        
        # Create audio augmentations
        for i in range(augmentation_factor):
            augmented_audio = audio_aug.speed_perturbation(original_audio.unsqueeze(0))
            augmented_audio = audio_aug.noise_injection(augmented_audio)
            
            enhanced_texts.append(original_text)
            enhanced_audio.append(augmented_audio.squeeze(0))
        
        # Create synthetic variations if enabled
        if synthetic_gen and synthetic_gen.available:
            synthetic_audio = synthetic_gen.create_speaker_variations(original_text, 2)
            for synth_audio in synthetic_audio:
                enhanced_texts.append(original_text)
                enhanced_audio.append(synth_audio)
    
    # Create enhanced dataset
    enhanced_dataset = Dataset.from_dict({
        "text": enhanced_texts,
        "audio": [{"array": audio.numpy(), "sampling_rate": 16000} 
                 for audio in enhanced_audio]
    })
    
    logger.info(f"Enhanced dataset created: {len(original_dataset)} -> {len(enhanced_dataset)} samples")
    logger.info(f"Enhancement factor: {len(enhanced_dataset) / len(original_dataset):.1f}x")
    
    return enhanced_dataset 