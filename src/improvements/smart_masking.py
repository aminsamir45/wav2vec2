"""
Smart Masking Strategies for wav2vec 2.0

WHY THIS IMPROVEMENT:
The original wav2vec 2.0 uses random span masking, which may not be optimal for 
low-resource scenarios. With limited labeled data, we need more intelligent 
masking that:

1. Focuses on linguistically meaningful units (phonemes, syllables)
2. Adapts masking ratio based on available data
3. Uses cross-lingual patterns for better generalization

EXPECTED BENEFITS:
- Better representation learning with limited data
- More robust features for downstream fine-tuning
- Improved generalization across different speakers/accents
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import librosa
from phonemizer import phonemize
import logging

logger = logging.getLogger(__name__)

class SmartMaskingStrategy:
    """Base class for intelligent masking strategies"""
    
    def __init__(self, mask_prob: float = 0.065, mask_length: int = 10):
        """
        Args:
            mask_prob: Probability of starting a masked span
            mask_length: Length of masked spans
        """
        self.mask_prob = mask_prob
        self.mask_length = mask_length
    
    def compute_mask_indices(self, shape: Tuple[int, int], **kwargs) -> torch.Tensor:
        """
        Compute mask indices for the given shape
        
        Args:
            shape: (batch_size, sequence_length)
            
        Returns:
            Boolean tensor indicating masked positions
        """
        raise NotImplementedError

class PhonemeAwareMasking(SmartMaskingStrategy):
    """
    Masking strategy that aligns with phoneme boundaries
    
    WHY: Random masking may split phonemes, making the contrastive task
    artificially difficult. Phoneme-aligned masking creates more realistic
    and learnable objectives.
    """
    
    def __init__(self, mask_prob: float = 0.065, mask_length: int = 10, 
                 language: str = "en-us"):
        super().__init__(mask_prob, mask_length)
        self.language = language
        logger.info(f"Initialized PhonemeAwareMasking for language: {language}")
    
    def get_phoneme_boundaries(self, text: str, audio_length: int, 
                             sample_rate: int = 16000) -> List[Tuple[int, int]]:
        """
        Estimate phoneme boundaries in the audio
        
        WHY: By masking complete phonemes rather than arbitrary spans,
        we create a more linguistically meaningful self-supervised task.
        
        Args:
            text: Transcription text
            audio_length: Length of audio in samples
            sample_rate: Audio sample rate
            
        Returns:
            List of (start, end) positions for each phoneme in frame indices
        """
        try:
            # Get phoneme sequence
            phonemes = phonemize(text, language=self.language, 
                               backend='espeak', strip=True, 
                               preserve_punctuation=False)
            phoneme_list = phonemes.split()
            
            # Estimate phoneme durations (simplified approach)
            # In practice, you'd use forced alignment tools like Montreal Forced Alignment
            audio_duration = audio_length / sample_rate
            avg_phoneme_duration = audio_duration / len(phoneme_list)
            
            # Convert to frame indices (assuming 20ms frames like wav2vec 2.0)
            frame_rate = 50  # 50 frames per second
            frames_per_phoneme = avg_phoneme_duration * frame_rate
            
            boundaries = []
            for i, phoneme in enumerate(phoneme_list):
                start_frame = int(i * frames_per_phoneme)
                end_frame = int((i + 1) * frames_per_phoneme)
                boundaries.append((start_frame, end_frame))
            
            return boundaries
            
        except Exception as e:
            logger.warning(f"Phoneme boundary estimation failed: {e}")
            # Fallback to uniform segmentation
            num_segments = max(1, audio_length // (sample_rate // 10))  # ~100ms segments
            segment_length = audio_length // num_segments
            return [(i * segment_length, (i + 1) * segment_length) 
                   for i in range(num_segments)]
    
    def compute_mask_indices(self, shape: Tuple[int, int], 
                           texts: Optional[List[str]] = None,
                           audio_lengths: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute phoneme-aware mask indices
        
        Args:
            shape: (batch_size, sequence_length)
            texts: List of transcription texts for each item in batch
            audio_lengths: List of audio lengths for each item in batch
            
        Returns:
            Boolean tensor indicating masked positions
        """
        batch_size, seq_len = shape
        mask_indices = torch.zeros(shape, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            if texts and audio_lengths:
                # Use phoneme information when available
                text = texts[batch_idx]
                audio_length = audio_lengths[batch_idx]
                
                try:
                    boundaries = self.get_phoneme_boundaries(text, audio_length)
                    
                    # Randomly select phonemes to mask
                    for start_frame, end_frame in boundaries:
                        if np.random.random() < self.mask_prob:
                            # Ensure boundaries are within sequence length
                            start_idx = min(start_frame, seq_len - 1)
                            end_idx = min(end_frame, seq_len)
                            if start_idx < end_idx:
                                mask_indices[batch_idx, start_idx:end_idx] = True
                                
                except Exception as e:
                    logger.warning(f"Falling back to random masking for batch {batch_idx}: {e}")
                    # Fallback to random masking for this sample
                    mask_indices[batch_idx] = self._random_mask(seq_len)
            else:
                # No text information available, use random masking
                mask_indices[batch_idx] = self._random_mask(seq_len)
        
        return mask_indices
    
    def _random_mask(self, seq_len: int) -> torch.Tensor:
        """Fallback random masking implementation"""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # Generate random mask spans
        num_masked = int(seq_len * self.mask_prob)
        if num_masked == 0:
            return mask
            
        for _ in range(num_masked // self.mask_length + 1):
            start = np.random.randint(0, max(1, seq_len - self.mask_length))
            end = min(start + self.mask_length, seq_len)
            mask[start:end] = True
            
        return mask

class AdaptiveMasking(SmartMaskingStrategy):
    """
    Adaptive masking that adjusts based on available training data
    
    WHY: With very limited labeled data (10min-1hr), we need to be more
    conservative with masking to ensure sufficient learning signal.
    More data allows for more aggressive masking.
    """
    
    def __init__(self, base_mask_prob: float = 0.065, 
                 min_mask_prob: float = 0.03, 
                 max_mask_prob: float = 0.1,
                 mask_length: int = 10):
        """
        Args:
            base_mask_prob: Default masking probability
            min_mask_prob: Minimum masking probability (for very low data)
            max_mask_prob: Maximum masking probability (for high data)
            mask_length: Length of masked spans
        """
        super().__init__(base_mask_prob, mask_length)
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
        self.current_mask_prob = base_mask_prob
        
        logger.info(f"Initialized AdaptiveMasking: prob range [{min_mask_prob}, {max_mask_prob}]")
    
    def adapt_masking_ratio(self, labeled_data_hours: float):
        """
        Adapt masking probability based on available labeled data
        
        WHY: Less labeled data requires more conservative masking to preserve
        learning signal. More labeled data allows aggressive masking for
        better representation learning.
        
        Args:
            labeled_data_hours: Amount of labeled training data in hours
        """
        if labeled_data_hours <= 0.17:  # 10 minutes or less
            self.current_mask_prob = self.min_mask_prob
            logger.info(f"Ultra-low resource ({labeled_data_hours:.2f}h): mask_prob = {self.current_mask_prob}")
        elif labeled_data_hours <= 1.0:  # 1 hour or less
            # Linear interpolation between min and base
            ratio = labeled_data_hours / 1.0
            self.current_mask_prob = self.min_mask_prob + ratio * (self.mask_prob - self.min_mask_prob)
            logger.info(f"Low resource ({labeled_data_hours:.2f}h): mask_prob = {self.current_mask_prob:.3f}")
        elif labeled_data_hours <= 10.0:  # 10 hours or less
            # Linear interpolation between base and max
            ratio = (labeled_data_hours - 1.0) / 9.0
            self.current_mask_prob = self.mask_prob + ratio * (self.max_mask_prob - self.mask_prob)
            logger.info(f"Medium resource ({labeled_data_hours:.2f}h): mask_prob = {self.current_mask_prob:.3f}")
        else:
            self.current_mask_prob = self.max_mask_prob
            logger.info(f"High resource ({labeled_data_hours:.2f}h): mask_prob = {self.current_mask_prob}")
    
    def compute_mask_indices(self, shape: Tuple[int, int], **kwargs) -> torch.Tensor:
        """
        Compute adaptive mask indices
        
        Uses the current adapted masking probability to generate masks.
        """
        batch_size, seq_len = shape
        mask_indices = torch.zeros(shape, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            mask_indices[batch_idx] = self._adaptive_mask(seq_len)
        
        return mask_indices
    
    def _adaptive_mask(self, seq_len: int) -> torch.Tensor:
        """Generate mask using current adaptive probability"""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # Use current adapted probability
        num_spans = int(seq_len * self.current_mask_prob / self.mask_length)
        
        for _ in range(num_spans):
            start = np.random.randint(0, max(1, seq_len - self.mask_length))
            end = min(start + self.mask_length, seq_len)
            mask[start:end] = True
            
        return mask

class CurriculumMasking(SmartMaskingStrategy):
    """
    Curriculum learning approach to masking
    
    WHY: Start with easier masking (shorter spans, lower probability) and
    gradually increase difficulty. This helps with convergence in low-resource
    scenarios where the model has limited data to learn from.
    """
    
    def __init__(self, initial_mask_prob: float = 0.03,
                 final_mask_prob: float = 0.065,
                 initial_mask_length: int = 5,
                 final_mask_length: int = 10,
                 curriculum_steps: int = 1000):
        """
        Args:
            initial_mask_prob: Starting masking probability (easier)
            final_mask_prob: Final masking probability (harder)
            initial_mask_length: Starting mask length (easier)
            final_mask_length: Final mask length (harder)
            curriculum_steps: Number of training steps to reach final difficulty
        """
        super().__init__(final_mask_prob, final_mask_length)
        self.initial_mask_prob = initial_mask_prob
        self.final_mask_prob = final_mask_prob
        self.initial_mask_length = initial_mask_length
        self.final_mask_length = final_mask_length
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
        
        logger.info(f"Initialized CurriculumMasking: "
                   f"prob {initial_mask_prob}->{final_mask_prob}, "
                   f"length {initial_mask_length}->{final_mask_length}")
    
    def update_curriculum(self, step: int):
        """
        Update curriculum based on training step
        
        WHY: Gradual increase in difficulty allows the model to build
        representations progressively, which is especially important
        when training data is limited.
        """
        self.current_step = step
        progress = min(1.0, step / self.curriculum_steps)
        
        # Linear curriculum progression
        self.mask_prob = (self.initial_mask_prob + 
                         progress * (self.final_mask_prob - self.initial_mask_prob))
        self.mask_length = int(self.initial_mask_length + 
                              progress * (self.final_mask_length - self.initial_mask_length))
        
        if step % 100 == 0:  # Log progress every 100 steps
            logger.info(f"Curriculum step {step}: mask_prob={self.mask_prob:.3f}, "
                       f"mask_length={self.mask_length}")
    
    def compute_mask_indices(self, shape: Tuple[int, int], **kwargs) -> torch.Tensor:
        """Compute curriculum-based mask indices"""
        batch_size, seq_len = shape
        mask_indices = torch.zeros(shape, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            mask_indices[batch_idx] = self._curriculum_mask(seq_len)
        
        return mask_indices
    
    def _curriculum_mask(self, seq_len: int) -> torch.Tensor:
        """Generate mask using current curriculum parameters"""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        num_spans = int(seq_len * self.mask_prob / self.mask_length)
        
        for _ in range(num_spans):
            start = np.random.randint(0, max(1, seq_len - self.mask_length))
            end = min(start + self.mask_length, seq_len)
            mask[start:end] = True
            
        return mask

def create_smart_masking_strategy(strategy_name: str, **kwargs) -> SmartMaskingStrategy:
    """
    Factory function to create masking strategies
    
    Args:
        strategy_name: Name of the strategy ('phoneme', 'adaptive', 'curriculum')
        **kwargs: Strategy-specific parameters
        
    Returns:
        Configured masking strategy
    """
    strategies = {
        'phoneme': PhonemeAwareMasking,
        'adaptive': AdaptiveMasking,
        'curriculum': CurriculumMasking,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(strategies.keys())}")
    
    logger.info(f"Creating smart masking strategy: {strategy_name}")
    return strategies[strategy_name](**kwargs) 