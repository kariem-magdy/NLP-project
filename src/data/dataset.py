# PyTorch Dataset for char-level labeling and transformer token classification
from torch.utils.data import Dataset
from typing import List, Tuple
import torch
from ..tokenizer import CharTokenizer
from ..preprocess import clean_line, normalize_text

class CharSequenceDataset(Dataset):
    """
    For BiLSTM+CRF: returns char indices and label indices.
    Train/dev files are expected to contain full diacritized lines.
    We use a simple mapping: each base-letter gets a diacritic label (like none/sukun/fatha/...)
    For simplicity labels are single-char diacritics; you may expand this mapping.
    """
    def __init__(self, file_path: str, char_tokenizer: CharTokenizer, label_map: dict, remove_diacritics_for_input=True):
        self.file_path = file_path
        self.char_tokenizer = char_tokenizer
        self.label_map = label_map  # {diacritic_char: idx}
        self.remove_diacritics = remove_diacritics_for_input
        self.samples = []
        self._load()

    def _extract_labels_from_text(self, line: str) -> Tuple[str, List[str]]:
        """
        Return (base_text_no_diacritics, labels_list_per_character)
        The label for each base character is the diacritic that follows it, or the special label '0' (no diacritic)
        Note: Arabic diacritics are combining marks. We assume train lines are already pre-normalized.
        """
        # naive implementation using unicode ranges: combining marks are 'tashkeel'
        # We'll iterate through characters and if a char is combining (i.e., diacritic), attach to previous base char.
        base_chars = []
        labels = []
        import unicodedata
        for ch in line:
            cat = unicodedata.category(ch)
            if cat.startswith('M'):  # mark, diacritic
                if not base_chars:
                    # stray diacritic -- ignore
                    continue
                labels[-1] = labels[-1] + ch
            else:
                base_chars.append(ch)
                labels.append('')  # empty label initially
        # map empty string to a canonical label like '_' for no diacritic
        labels = [lab if lab != '' else '_' for lab in labels]
        base_text = ''.join(base_chars)
        return base_text, labels

    def _load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = clean_line(line)
                if not line:
                    continue
                base_text, labels = self._extract_labels_from_text(line)
                if self.remove_diacritics:
                    base_text = normalize_text(base_text, remove_diacritics=True)
                self.samples.append((base_text, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_text, labels = self.samples[idx]
        char_idxs = self.char_tokenizer.text_to_sequence(base_text)
        label_idxs = [self.label_map.get(l, self.label_map['_']) for l in labels]
        return {
            'chars': torch.tensor(char_idxs, dtype=torch.long),
            'labels': torch.tensor(label_idxs, dtype=torch.long),
            'raw': base_text,
            'label_strs': labels
        }
