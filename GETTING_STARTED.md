# Getting Started

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using setup.py
pip install -e .
```

### 2. Download Original Models and Data

```bash
# Download pre-trained models and LibriSpeech data
python scripts/setup_original_wav2vec2.py
```

### 3. Run Baseline Experiments

```bash
# Test the baseline implementation
python src/baseline/wav2vec2_baseline.py
```

## 📁 Project Structure

```
wav2vec2/
├── PROJECT_PLAN.md           # Project overview and goals
├── GETTING_STARTED.md        # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── src/
│   ├── __init__.py
│   └── baseline/
│       └── wav2vec2_baseline.py   # Baseline implementation
├── scripts/
│   └── setup_original_wav2vec2.py # Setup script
└── models/
    └── pretrained/           # Downloaded models (created by setup)
```

## 🔧 Original wav2vec 2.0 Code Access

The original wav2vec 2.0 code is available at:
- **Paper**: https://arxiv.org/pdf/2006.11477
- **Code**: https://github.com/pytorch/fairseq (fairseq implementation)
- **Models**: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec

Our baseline uses the HuggingFace transformers implementation, which is more accessible and easier to modify.

## 🎯 Next Steps

1. **Run baseline**: Verify everything works with the baseline
2. **Check results**: Compare your baseline results with paper results
3. **Start experimenting**: Begin implementing improvements in `src/improvements/`

## 💻 Commands to Run

```bash
# Install everything
pip install -r requirements.txt

# Download models and data  
python scripts/setup_original_wav2vec2.py

# Run baseline test
python src/baseline/wav2vec2_baseline.py
```

That's it! You're ready to start experimenting with wav2vec 2.0 improvements. 