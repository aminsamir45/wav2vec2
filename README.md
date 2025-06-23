# Efficient wav2vec 2.0 for Ultra-Low Resource Speech Recognition

A research project implementing architectural improvements to wav2vec 2.0 for enhanced performance in ultra-low resource scenarios (10 minutes to 1 hour of labeled speech data).

## 🎯 Project Overview

This project addresses a critical limitation of current speech recognition systems: they require thousands of hours of transcribed speech to reach acceptable performance. Our improvements enable effective speech recognition with as little as 10 minutes of labeled data.

**Key Research Question**: How can we modify wav2vec 2.0 to achieve better performance when training data is extremely limited?

## 📊 Background: Original wav2vec 2.0

From the [original paper](https://arxiv.org/pdf/2006.11477):
- **Achieves 4.8/8.2 WER** with only 10 minutes of labeled data
- **Architecture**: CNN feature encoder + Transformer + Quantization module
- **Training**: Self-supervised pre-training + supervised fine-tuning
- **Challenge**: Random masking and full fine-tuning may not be optimal for ultra-low resource scenarios

## 🔬 Our Improvements

### 1. Smart Masking Strategies (`src/improvements/smart_masking.py`)

**WHY**: Random span masking may not be optimal for limited data scenarios.

**Implementations**:
- **Phoneme-Aware Masking**: Aligns masks with linguistic boundaries for more meaningful learning objectives
- **Adaptive Masking**: Adjusts masking probability based on available data (conservative with less data)
- **Curriculum Masking**: Starts with easier masking and gradually increases difficulty

```python
from src.improvements.smart_masking import create_smart_masking_strategy

# Adaptive masking for ultra-low resource
masking = create_smart_masking_strategy("adaptive", min_mask_prob=0.03, max_mask_prob=0.1)
masking.adapt_masking_ratio(labeled_data_hours=0.17)  # 10 minutes
```

### 2. Parameter-Efficient Fine-tuning (`src/improvements/parameter_efficient.py`)

**WHY**: Full model fine-tuning with limited data leads to overfitting.

**Implementations**:
- **LoRA (Low-Rank Adaptation)**: Adds small trainable matrices instead of updating full weights
- **Adapter Layers**: Inserts lightweight bottleneck layers between transformer blocks
- **Progressive Unfreezing**: Gradually unfreezes layers during training
- **Hybrid Approaches**: Combines multiple techniques

```python
from src.improvements.parameter_efficient import apply_parameter_efficient_adaptation

# Apply LoRA with rank 16 (reduces trainable params by ~99%)
efficient_model = apply_parameter_efficient_adaptation(
    base_model, method="lora", lora_rank=16
)
```

### 3. Enhanced Data Utilization (`src/improvements/data_augmentation.py`)

**WHY**: Every sample is precious with limited data - we need intelligent augmentation.

**Implementations**:
- **Audio Augmentation**: Speed perturbation, noise injection, pitch shifting
- **Synthetic Data Generation**: TTS-based data creation from text transcripts
- **Cross-lingual Transfer**: Leverage high-resource languages
- **Self-training**: Use model predictions to expand training data

### 4. Systematic Evaluation (`src/improvements/experiment_runner.py`)

**WHY**: Need rigorous validation to prove improvements work.

**Features**:
- Baseline vs improved model comparisons
- Ablation studies for each improvement
- Evaluation on 10min/1hr/10hr data scenarios
- Automated result reporting and analysis

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd wav2vec2

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models and data
python scripts/setup_original_wav2vec2.py
```

### Quick Start

```bash
# 1. Run baseline experiments
python src/baseline/wav2vec2_baseline.py

# 2. Test individual improvements
python -c "
from src.improvements.smart_masking import create_smart_masking_strategy
masking = create_smart_masking_strategy('adaptive')
print('Adaptive masking created successfully!')
"

# 3. Run full experimental suite
python src/improvements/experiment_runner.py
```

## 📈 Expected Results

Based on our research and similar work in the field:

| Method | 10min WER | 1hr WER | Trainable Params | Key Benefit |
|--------|-----------|---------|------------------|-------------|
| Baseline | ~12-15% | ~8-10% | 100% (95M) | Strong pre-trained features |
| + Adaptive Masking | ~11-14% | ~7-9% | 100% | Better representation learning |
| + LoRA (rank 16) | ~10-13% | ~6-8% | ~1% (1M) | Reduced overfitting |
| + Data Augmentation | ~9-12% | ~5-7% | Variable | More diverse training data |
| **All Combined** | **~8-11%** | **~4-6%** | **~1%** | **Best of all worlds** |

## 🔧 Technical Details

### Architecture Modifications

1. **Smart Masking Integration**: Replace random masking in pre-training
2. **Parameter-Efficient Layers**: Add LoRA/Adapter layers to transformer blocks
3. **Enhanced Data Pipeline**: Integrate augmentation during training
4. **Evaluation Framework**: Systematic comparison infrastructure

### Key Design Decisions

- **Compatibility**: All improvements work with existing wav2vec 2.0 checkpoints
- **Modularity**: Each improvement can be used independently or combined
- **Efficiency**: Focus on computational efficiency for practical deployment
- **Reproducibility**: Comprehensive logging and deterministic experiments

## 📊 File Structure

```
wav2vec2/
├── PROJECT_PLAN.md              # Detailed research plan
├── GETTING_STARTED.md           # Quick setup guide
├── requirements.txt             # Dependencies
├── src/
│   ├── baseline/
│   │   └── wav2vec2_baseline.py # Original model implementation
│   └── improvements/
│       ├── smart_masking.py     # Intelligent masking strategies
│       ├── parameter_efficient.py # LoRA, adapters, progressive unfreezing
│       ├── data_augmentation.py # Data enhancement techniques
│       └── experiment_runner.py # Systematic evaluation framework
├── scripts/
│   └── setup_original_wav2vec2.py # Download models and data
└── models/
    └── pretrained/              # Downloaded model checkpoints
```

## 🎯 Research Contributions

1. **Smart Masking for Low-Resource ASR**: Novel masking strategies tailored for limited data scenarios
2. **Parameter-Efficient Adaptation**: Systematic application of LoRA/adapters to speech recognition
3. **Comprehensive Evaluation**: Rigorous experimental framework for ultra-low resource scenarios
4. **Practical Implementation**: Production-ready code with clear documentation

## 📚 Related Work

- **wav2vec 2.0**: [Baevski et al., 2020](https://arxiv.org/pdf/2006.11477)
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **Adapters**: [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)
- **Low-Resource ASR**: [Kahn et al., 2019](https://arxiv.org/abs/1904.03240)

## 🔬 Experimental Validation

Run the full experimental suite to validate improvements:

```bash
python src/improvements/experiment_runner.py
```

This will:
1. Create 10min/1hr/10hr LibriSpeech subsets
2. Evaluate baseline performance
3. Test each improvement individually
4. Test promising combinations
5. Generate comprehensive results report

## 🎯 Key Findings (Expected)

1. **Smart masking** improves sample efficiency by 5-8%
2. **Parameter-efficient fine-tuning** reduces overfitting significantly
3. **Combined approaches** achieve 15-20% improvement over baseline
4. **Computational efficiency** maintained or improved

## 💡 Future Work

- **Multi-language validation**: Test on low-resource languages beyond English
- **Real-world deployment**: Edge device optimization and streaming inference
- **Domain adaptation**: Extend to medical, legal, and other specialized domains
- **Semi-supervised learning**: Integrate with self-training approaches

## 🤝 Contributing

This is a research project demonstrating technical skills. The implementations are designed to be:
- **Educational**: Clear documentation of WHY each improvement matters
- **Reproducible**: Systematic experimental validation
- **Practical**: Real-world applicable techniques
- **Extensible**: Easy to add new improvements

## 📄 License

This project is for educational and research purposes, demonstrating advanced machine learning and speech recognition techniques.

---

**Research Focus**: Ultra-low resource speech recognition using intelligent wav2vec 2.0 modifications

**Technical Skills Demonstrated**: Deep learning, speech processing, parameter-efficient training, experimental validation, research methodology