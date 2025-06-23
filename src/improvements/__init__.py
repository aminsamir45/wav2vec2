"""
Improvements to wav2vec 2.0 for Ultra-Low Resource Speech Recognition

This package contains architectural and training improvements designed to enhance
wav2vec 2.0's performance in scenarios with very limited labeled data (10min-1hr).

Improvements implemented:
1. Smart Masking Strategies - More intelligent masking patterns
2. Parameter-Efficient Fine-tuning - Reduce trainable parameters
3. Enhanced Data Utilization - Better use of limited data
4. Lightweight Architecture Variants - Computationally efficient modifications

Each improvement is designed to address specific limitations of the original
wav2vec 2.0 when working with ultra-low resource scenarios.
"""

__version__ = "0.1.0" 