# src/preprocess.py
"""
Preprocessing module for Arabic diacritization dataset.

Implements:
 - detect_encoding(path)
 - clean_line(line)
 - extract_labels(line)
 - normalize_text(line, options)
 - strip_diacritics(line)
 - build_char_vocab(texts, min_freq=1)
 - build_label_map(labels)
 - save_json / load_json
 - make_dataset(file_path, char2idx, label2idx, strip_input_diacritics=True)
 - collate_fn(batch)
 - example_usage()
 
Dependencies: regex, unicodedata, json, os, logging, typing, collections, torch
"""

from typing import Tuple, List, Dict, Iterable, Optional, Any
import unicodedata
import regex as re
import json
import os
import logging
from collections import Counter, OrderedDict
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")

# Constants / tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL = -100  # label index used for padded positions in loss

# Regex helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]")  # zero-width & rtl marks

# Arabic code block test function
def _is_arabic_char(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        (0x0600 <= o <= 0x06FF) or
        (0x0750 <= o <= 0x077F) or
        (0x08A0 <= o <= 0x08FF) or
        (0xFB50 <= o <= 0xFDFF) or
        (0xFE70 <= o <= 0xFEFF) or
        (0x0610 <= o <= 0x061A) or
        (0x064B <= o <= 0x065F)
    )

def detect_encoding(path: str) -> str:
    """Very lightweight encoding check: attempts to read as utf-8."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            _ = f.read(4096)
        return "utf-8"
    except UnicodeDecodeError as e:
        logger.error("File %s is not UTF-8 encoded (error: %s).", path, e)
        raise

def clean_line(line: str, remove_urls: bool = True, remove_emails: bool = True,
               remove_zero_width: bool = True, normalize_spaces: bool = True) -> str:
    """Clean a single line: remove URLs, emails, zero-width chars."""
    s = line
    if remove_urls:
        s = _URL_RE.sub(" ", s)
    if remove_emails:
        s = _EMAIL_RE.sub(" ", s)
    if remove_zero_width:
        s = _ZERO_WIDTH_RE.sub("", s)
    # remove control characters (except newline which is handled at file level)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    if normalize_spaces:
        s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_diacritics(line: str) -> str:
    """Remove Arabic combining marks (diacritics) from line."""
    return "".join(ch for ch in line if unicodedata.category(ch)[0] != "M")

def extract_labels(line: str) -> Tuple[str, List[str]]:
    """
    Given a fully diacritized line, returns:
      - base_text: str of base characters (characters that are NOT combining marks)
      - labels: list of strings (label per base character).
    """
    base_chars: List[str] = []
    labels: List[str] = []
    pending_marks: List[str] = []

    for ch in line:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):  # combining mark
            pending_marks.append(ch)
        else:
            # This is a base character. 
            if base_chars: 
                # If there were pending marks, they belong to the PREVIOUS base character.
                if pending_marks:
                    prev_label = labels[-1]
                    if prev_label == "_":
                        labels[-1] = "".join(pending_marks)
                    else:
                        labels[-1] = prev_label + "".join(pending_marks)
                    pending_marks = []
            
            # start new base char
            base_chars.append(ch)
            labels.append("_")  # default label

    # Handle trailing marks (at end of line)
    if pending_marks:
        if base_chars:
            prev_label = labels[-1]
            if prev_label == "_":
                labels[-1] = "".join(pending_marks)
            else:
                labels[-1] = prev_label + "".join(pending_marks)
        else:
            logger.warning(
                "Found trailing/stray combining marks with no base char: %r -> ignored",
                "".join(pending_marks),
            )

    base_text = "".join(base_chars)
    return base_text, labels

def normalize_text(line: str, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Normalize the Arabic text.
    NOTE: Should be applied BEFORE label extraction to ensure alignment.
    """
    if options is None:
        options = {}
    normalize_hamza = options.get("normalize_hamza", True)
    remove_tatweel = options.get("remove_tatweel", True)
    lower_latin = options.get("lower_latin", True)
    remove_punct = options.get("remove_punctuation", False)

    s = line
    # normalize to NFKC for stability
    s = unicodedata.normalize("NFKC", s)
    
    if remove_tatweel:
        s = s.replace("\u0640", "")

    if normalize_hamza:
        s = s.replace("\u0622", "\u0627")  # آ -> ا
        s = s.replace("\u0623", "\u0627")  # أ -> ا
        s = s.replace("\u0625", "\u0627")  # إ -> ا

    if lower_latin:
        s = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), s)

    if remove_punct:
        # Keep letters (L), numbers (N), and marks (M) - critical to keep diacritics!
        s = re.sub(r"[^\p{L}\p{N}\p{M}\s]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_char_vocab(texts: Iterable[str], min_freq: int = 1) -> Tuple[Dict[str,int], Dict[int,str]]:
    """Build char2idx and idx2char."""
    counter = Counter()
    for t in texts:
        counter.update(list(t))
    
    items = [(ch, freq) for ch, freq in counter.items() if freq >= min_freq]
    # sort by freq desc, then codepoint ascending
    items.sort(key=lambda x: (-x[1], ord(x[0])))
    
    char2idx: Dict[str,int] = OrderedDict()
    char2idx[PAD_TOKEN] = 0
    char2idx[UNK_TOKEN] = 1
    idx = 2
    for ch, _ in items:
        if ch in char2idx:
            continue
        char2idx[ch] = idx
        idx += 1
    idx2char = {i: c for c, i in char2idx.items()}
    return dict(char2idx), dict(idx2char)

def build_label_map(labels: Iterable[List[str]]) -> Tuple[Dict[str,int], Dict[int,str]]:
    """Build label2idx and idx2label."""
    s = set()
    for lablist in labels:
        s.update(lablist)
    if "_" not in s:
        s.add("_")
    
    other_labels = sorted([l for l in s if l != "_"], key=lambda x: (len(x), x))
    label2idx = OrderedDict()
    label2idx["_"] = 0
    for i, lab in enumerate(other_labels, start=1):
        label2idx[lab] = i
    idx2label = {i: l for l, i in label2idx.items()}
    return dict(label2idx), dict(idx2label)

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class DiacritizationDataset(Dataset):
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        return {
            "chars": torch.tensor(e["char_ids"], dtype=torch.long),
            "labels": torch.tensor(e["label_ids"], dtype=torch.long),
            "raw": e["base"],
            "label_strs": e["labels"]
        }

def collate_fn(batch: List[Dict[str, Any]], pad_idx: int = 0):
    chars = [item["chars"] for item in batch]
    labels = [item["labels"] for item in batch]
    raws = [item["raw"] for item in batch]
    label_strs = [item["label_strs"] for item in batch]

    padded_chars = pad_sequence(chars, batch_first=True, padding_value=pad_idx)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)
    mask = (padded_chars != pad_idx)
    
    return {
        "chars": padded_chars,
        "labels": padded_labels,
        "mask": mask,
        "raws": raws,
        "label_strs": label_strs
    }

def make_dataset(file_path: str, char2idx: Dict[str,int], label2idx: Dict[str,int],
                 strip_input_diacritics: bool = True, 
                 normalization_options: Optional[Dict[str, Any]] = None) -> DiacritizationDataset:
    """
    Corrected logic: Normalize line -> Extract Labels -> Map to IDs.
    """
    # 1. Handle default mutable argument safely
    if normalization_options is None:
        normalization_options = {
            "normalize_hamza": True, 
            "remove_tatweel": True, 
            "lower_latin": True, 
            "remove_punctuation": True
        }

    entries = []
    detect_encoding(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            line = clean_line(line)
            if len(line.strip()) == 0:
                continue

            # 2. FIX: Normalize BEFORE extracting labels.
            # This ensures that if characters are removed (e.g. tatweel), 
            # the labels are extracted from the final version of the text.
            if normalization_options:
                line = normalize_text(line, normalization_options)

            base_text, labels = extract_labels(line)
            
            if len(base_text) == 0:
                logger.warning("Line %d -> base_text empty after extract; skipping", i)
                continue
            
            # Note: extract_labels returns base_text already stripped of diacritics.
            # We don't need to call strip_diacritics(base_text) again.
            
            char_ids = [char2idx.get(ch, char2idx.get(UNK_TOKEN)) for ch in base_text]
            
            # 3. Validation
            if len(labels) != len(base_text):
                logger.error("Line %d: Alignment Error. Base len %d vs Label len %d. Skipping.", i, len(base_text), len(labels))
                continue

            label_ids = [label2idx.get(lab, label2idx.get("_")) for lab in labels]
            
            entries.append({
                "raw": base_text,
                "base": base_text,
                "labels": labels,
                "char_ids": char_ids,
                "label_ids": label_ids
            })

    logger.info("Built dataset from %s: %d examples", file_path, len(entries))
    return DiacritizationDataset(entries)

def example_usage(input_path: str = "data/train.txt",
                  out_dir: str = "outputs/processed",
                  min_freq: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    
    # Defaults for consistency
    norm_opts = {"normalize_hamza": True, "remove_tatweel": True}

    detect_encoding(input_path)
    base_texts = []
    label_lists = []
    
    # Pass 1: Build Vocab (Apply same normalization as make_dataset!)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            line = clean_line(line)
            if not line: continue
            
            # Important: Normalize here too, otherwise vocab won't match dataset
            line = normalize_text(line, norm_opts)
            
            base, labs = extract_labels(line)
            if not base: continue
            
            base_texts.append(base)
            label_lists.append(labs)
            
    char2idx, idx2char = build_char_vocab(base_texts, min_freq=min_freq)
    label2idx, idx2label = build_label_map(label_lists)
    
    save_json(char2idx, os.path.join(out_dir, "char2idx.json"))
    save_json(idx2char, os.path.join(out_dir, "idx2char.json"))
    save_json(label2idx, os.path.join(out_dir, "label2idx.json"))
    save_json(idx2label, os.path.join(out_dir, "idx2label.json"))
    
    logger.info("Saved vocab to %s", out_dir)
    return os.path.join(out_dir, "char2idx.json"), os.path.join(out_dir, "label2idx.json")

if __name__ == "__main__": 
    example_usage()