# character-level tokenizer and label mapping
from collections import Counter
from typing import List, Dict
import json
import os

class CharTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.char2idx = {self.PAD:0, self.UNK:1}
        self.idx2char = {0:self.PAD, 1:self.UNK}
        self.fitted = False

    def fit_on_texts(self, texts: List[str], min_freq: int = 1):
        counter = Counter()
        for t in texts:
            for ch in t:
                counter[ch] += 1
        idx = len(self.char2idx)
        for ch, c in counter.items():
            if c >= min_freq and ch not in self.char2idx:
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
                idx += 1
        self.fitted = True

    def text_to_sequence(self, text: str):
        return [self.char2idx.get(ch, self.char2idx[self.UNK]) for ch in text]

    def sequence_to_text(self, seq):
        return ''.join(self.idx2char.get(i, self.UNK) for i in seq)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.char2idx, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)
        self.idx2char = {v:k for k,v in self.char2idx.items()}
        self.fitted = True
