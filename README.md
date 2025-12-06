Here is a professional **README.md** file tailored to your project. It focuses on local installation and usage, omitting Docker as requested.

You can copy the content below directly into your `README.md` file.

-----

````markdown
# Arabic Text Diacritization System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Natural Language Processing (NLP) pipeline for automatic Arabic diacritization. This project implements two state-of-the-art approaches: a feature-rich **BiLSTM-CRF** baseline and a fine-tuned **Transformer (AraBERT)** model. It is designed for high performance, modularity, and ease of deployment.

## 🚀 Key Features

### 1. Advanced BiLSTM-CRF Architecture
A robust sequence labeling model incorporating multiple linguistic features to capture both morphological and semantic context:
* **Character Embeddings:** Dense vector representations for individual Arabic characters.
* **Word Embeddings:** Trainable embeddings aligned with character sequences.
* **FastText Integration:** Support for pre-trained `wiki.ar.vec` embeddings to handle out-of-vocabulary words.
* **Syntactic Features:** Sentence-level context capture using **Bag-of-Words (BoW)** and **TF-IDF** vectors projected into the sequence space.
* **CRF Layer:** Conditional Random Field output layer to model valid diacritic transitions and dependencies.

### 2. Transformer Fine-Tuning
* Direct fine-tuning of **AraBERT** (or other BERT-based models) for token classification.
* Handles sub-token alignment automatically.
* Optimized training loop using Hugging Face's `Trainer` API.

### 3. Production Ready
* **Flask API:** A lightweight web interface for real-time demonstration.
* **Configurable:** Centralized configuration for easy hyperparameter tuning.
* **Inference Engine:** optimized script for processing raw text.

---

## 📂 Project Structure

```text
├── build_vocab.py           # Preprocessing script to generate vocabularies & feature vectorizers
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script (Linux/Git Bash)
├── data/                    # Dataset directory
│   ├── train.txt            # Training data (UTF-8 text)
│   ├── val.txt              # Validation data
│   └── fasttext_wiki.ar.vec # (Optional) Pre-trained FastText vectors
├── outputs/                 # Artifacts directory
│   ├── models/              # Saved model checkpoints (.pt)
│   ├── logs/                # Training logs
│   └── processed/           # Vocabularies (.json) and feature pickles (.pkl)
├── scripts/                 # Execution shell scripts
└── src/                     # Source code
    ├── app/                 # Flask web application
    ├── config.py            # Central configuration (Paths, Hyperparams, Flags)
    ├── features.py          # Feature extraction logic (BoW, TF-IDF, FastText)
    ├── preprocess.py        # Text cleaning, normalization, and label extraction
    ├── data/                # Dataset loading and batch collation
    ├── models/              # Model architectures (BiLSTM-CRF, Transformer)
    ├── train/               # Training loops
    ├── infer/               # Inference engine
    └── eval/                # Evaluation metrics (DER)
````

-----

## 🛠️ Installation

### 1\. Prerequisites

  * Python 3.8+
  * CUDA-compatible GPU (Recommended)

### 2\. Setup

Clone the repository and install dependencies:

```bash
git clone [https://github.com/kariem-magdy/NLP-project.git](https://github.com/kariem-magdy/NLP-project.git)
cd "NLP project"
pip install -r requirements.txt
```

*(Note: If you are on Windows/Linux, you may need to install `ffmpeg` and `libsndfile1` if any audio libraries are triggered, though this project focuses on text).*

-----

## 📊 Data Preparation

Before training, you must process the raw text files to generate vocabulary mappings (`char2idx`, `label2idx`) and train the feature vectorizers (BoW, TF-IDF).

1.  Ensure your data files (`train.txt`, `val.txt`) are in the `data/` directory.
2.  (Optional) Download `wiki.ar.vec` from [FastText](https://fasttext.cc/docs/en/pretrained-vectors.html) and place it in `data/` as `fasttext_wiki.ar.vec` for enhanced performance.
3.  Run the build script:

<!-- end list -->

```bash
python build_vocab.py
```

*This will generate all necessary artifacts in `outputs/processed/`.*

-----

## 🧠 Usage

### Training

You can toggle features and adjust hyperparameters (e.g., `batch_size`, `use_fasttext`, `use_bow`) in `src/config.py`.

**To train the BiLSTM-CRF model:**

```bash
python -m src.train.train_bilstm
```

  * **Checkpoint:** Saved to `outputs/models/best_bilstm.pt`
  * **Metric:** Monitors Diacritic Error Rate (DER).

**To train the Transformer model:**

```bash
python -m src.train.train_transformer
```

### Inference (CLI)

To diacritize a single sentence or verify the model:

```bash
python -m src.infer.infer
```

*You can modify the test sentence inside `src/infer/infer.py`.*

### Web Demo (Flask)

Launch the Flask application to test the model via a browser:

```bash
python -m src.app.app
```

Access the demo at `http://localhost:5000`.

-----

## 📉 Evaluation

The system evaluates performance using the **Diacritic Error Rate (DER)**, which calculates the percentage of incorrectly predicted diacritics relative to the total number of diacritized characters (excluding padding).

$$\text{DER} = \left( \frac{\text{Incorrect Predictions}}{\text{Total Valid Characters}} \right) \times 100 \%$$

-----

## ⚙️ Configuration (`src/config.py`)

You can control the entire pipeline by editing `src/config.py`. Key flags include:

```python
# Feature Flags
use_word_emb = True       # Enable trainable word embeddings
use_fasttext = False      # Enable pre-trained FastText (requires .vec file)
use_bow = True            # Enable Bag of Words features
use_tfidf = True          # Enable TF-IDF features

# Hyperparameters
char_emb_dim = 128
lstm_hidden = 256
batch_size = 32
epochs = 20
lr = 1e-3
```

-----

## 📜 License

This project is licensed under the MIT License.

```
```