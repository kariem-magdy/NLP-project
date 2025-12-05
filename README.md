# Arabic Diacritizer (PyTorch + HuggingFace + Flask)

## Overview
Starter repo for Arabic diacritization:
- Char-level BiLSTM + CRF baseline (PyTorch)
- Transformer-based fine-tuning (Hugging Face)
- Flask demo for single-sentence inference using GPU if available.

## Quick start
1. Create venv & install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
