# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from ..preprocess import parse_file_to_entries, UNK_TOKEN

class DiacritizationDataset(Dataset):
    """
    PyTorch Dataset that delegates processing to src.preprocess.
    """
    def __init__(self, 
                 file_path: str, 
                 char2idx: Dict[str, int], 
                 label2idx: Dict[str, int],
                 normalize_options: Optional[Dict] = None):
        
        self.char2idx = char2idx
        self.label2idx = label2idx
        
        # Default Normalization options matching filtering.py
        if normalize_options is None:
            normalize_options = {
                "normalize_hamza": True, 
                "remove_tatweel": True, 
                "lower_latin": True, 
                "remove_punctuation": True
            }

        # Delegate parsing to the robust preprocess module
        self.entries = parse_file_to_entries(
            file_path=file_path,
            char2idx=char2idx,
            label2idx=label2idx,
            strip_input_diacritics=True,
            normalization_options=normalize_options
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return {
            "chars": torch.tensor(e["char_ids"], dtype=torch.long),
            "labels": torch.tensor(e["label_ids"], dtype=torch.long),
            "raw": e["raw"],
            "label_strs": e["labels"]
        }