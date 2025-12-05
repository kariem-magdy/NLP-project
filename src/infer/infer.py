# src/infer/infer.py
import torch
import os
import unicodedata
from ..models.bilstm_crf import BiLSTMCRF
from ..preprocess import clean_line, normalize_text, load_json, pad_sequence
from ..config import cfg

class DiacriticPredictor:
    def __init__(self, model_path=None, vocab_dir=None):
        self.device = torch.device(cfg.device)
        
        # 1. Resolve Paths
        if vocab_dir is None:
            vocab_dir = os.path.join(cfg.outputs_dir, "processed")
        if model_path is None:
            model_path = os.path.join(cfg.models_dir, "best_bilstm.pt")

        # 2. Load Vocabs
        self.char2idx = load_json(os.path.join(vocab_dir, "char2idx.json"))
        self.label2idx = load_json(os.path.join(vocab_dir, "label2idx.json"))
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.idx2char = {v: k for k, v in self.char2idx.items()}

        # 3. Load Model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle cases where checkpoint might be full dict or just state_dict
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        self.model = BiLSTMCRF(
            vocab_size=len(self.char2idx),
            char_emb_dim=cfg.char_emb_dim,
            lstm_hidden=cfg.lstm_hidden,
            num_labels=len(self.label2idx),
            num_layers=cfg.lstm_layers,
            dropout=cfg.bilstm_dropout,
            pad_idx=0
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        # Clean & Normalize (Must match training logic!)
        text = clean_line(text)
        text = normalize_text(text, {
            "normalize_hamza": True, 
            "remove_tatweel": True, 
            "lower_latin": True, 
            "remove_punctuation": True # Caution: Verify if you want punct removed in final output
        })
        return text

    def predict(self, text):
        """
        Diacritizes a raw input string.
        """
        # 1. Preprocess
        clean_text = self.preprocess_text(text)
        if not clean_text:
            return ""

        # 2. Vectorize
        char_ids = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in clean_text]
        input_tensor = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        mask = torch.ones_like(input_tensor, dtype=torch.bool).to(self.device)

        # 3. Inference
        with torch.no_grad():
            # Model returns list of lists of label indices
            pred_ids = self.model(input_tensor, mask=mask)[0]

        # 4. Decode & Merge
        output_chars = []
        for char, p_id in zip(clean_text, pred_ids):
            diacritic = self.idx2label.get(p_id, "_")
            output_chars.append(char)
            if diacritic != "_":
                output_chars.append(diacritic)

        return "".join(output_chars)