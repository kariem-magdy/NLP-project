# src/models/transformer_finetune.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Any

class TransformerTokenClassifier:
    """
    Simple thin wrapper to use HF AutoModelForTokenClassification for inference & saving/loading.
    Training is done with Trainer in train_transformer.py
    """
    def __init__(self, model_name: str, label_list: List[str], device: str = "cpu"):
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
        self.model.eval()

    def predict_on_sentences(self, sentences: List[str]) -> Tuple[List[List[int]], Any]:
        """
        Returns (predictions, encoding) where predictions is list-of-list label ids (padded).
        The consumer must map ids -> labels using label_list.
        """
        enc = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, is_split_into_words=False)
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, C)
            preds = logits.argmax(dim=-1).cpu().tolist()
        return preds, enc
