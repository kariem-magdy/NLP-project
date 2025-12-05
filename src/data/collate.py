# src/data/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(batch: List[Dict], pad_idx: int = 0, pad_label: int = -100):
    """
    Pads sequences for training.
    pad_idx: padding for character IDs (usually 0)
    pad_label: padding for labels (usually -100 to ignore in CrossEntropy)
    """
    # Extract
    chars = [item["chars"] for item in batch]
    labels = [item["labels"] for item in batch]
    raws = [item["raw"] for item in batch]
    label_strs = [item["label_strs"] for item in batch]

    # Pad
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=pad_idx)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_label)
    
    # Create Mask (1 for real token, 0 for pad)
    mask = (chars_padded != pad_idx)

    return {
        "chars": chars_padded,
        "labels": labels_padded,
        "mask": mask,
        "raws": raws,
        "label_strs": label_strs
    }