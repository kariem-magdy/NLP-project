# wrapper to fine-tune a transformer for token classification (we'll align characters to tokens)
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np

class TransformerTokenClassifier:
    def __init__(self, model_name, label_list, device):
        self.model_name = model_name
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=self.num_labels).to(self.device)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)

    def predict_on_sentences(self, sentences: list):
        # sentences: list of strings (no diacritics)
        enc = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, is_split_into_words=False)
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, num_labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        # Map token-level predictions to characters is dataset-specific; here we return token predictions
        return preds, enc
