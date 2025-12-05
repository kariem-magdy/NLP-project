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
PAD_LABEL = -100  # label index used for padded positions in loss (compatible with PyTorch loss APIs)

# Regex helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]")  # zero-width & rtl marks
_COMBINING_RE = re.compile(r"\p{M}+")  # one or more combining marks (unicode property 'M')

# Arabic code block test function (used to decide if char is "Arabic" letter or mark)
def _is_arabic_char(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    # include Arabic blocks + presentation forms + Arabic diacritic ranges
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
    """
    Very lightweight encoding check: attempts to read as utf-8.
    Returns 'utf-8' if ok, otherwise raises.
    (If needed convert externally and re-run.)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            _ = f.read(4096)
        return "utf-8"
    except UnicodeDecodeError as e:
        logger.error("File %s is not UTF-8 encoded (error: %s).", path, e)
        raise

def clean_line(line: str, remove_urls: bool = True, remove_emails: bool = True,
               remove_zero_width: bool = True, normalize_spaces: bool = True) -> str:
    """
    Clean a single line:
      - remove URLs and emails
      - remove zero-width and RTL/LTR marks
      - normalize various whitespace sequences to single space
      - strip surrounding whitespace
    This function intentionally does NOT remove numeric citations or punctuation by default.
    """
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
    """
    Remove Arabic combining marks (diacritics) from line.
    Uses Unicode category checks starting with 'M'.
    """
    return "".join(ch for ch in line if unicodedata.category(ch)[0] != "M")

def extract_labels(line: str) -> Tuple[str, List[str]]:
    """
    Given a fully diacritized line, returns:
      - base_text: str of base characters (characters that are NOT combining marks)
      - labels: list of strings (label per base character). If a base char has no diacritic the label is '_'.

    Behavior / edge cases:
      - Leading combining marks (no preceding base char): they are logged and ignored.
      - Multiple combining marks attached to one base char are concatenated into a single label string (e.g., shadda+fatha).
      - Non-Arabic base characters (Latin letters, digits, punctuation) are included in base_text and will get labels '_' (unless diacritics follow them).
      - Trailing combining marks (at end of the line): if there is at least one base char, they are attached to the last base char's label.
        If there are no base chars at all, they are logged and ignored (same as leading marks).
    """
    base_chars: List[str] = []
    labels: List[str] = []
    pending_marks: List[str] = []

    for ch in line:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):  # combining mark
            pending_marks.append(ch)
        else:
            # This is a base character. If there were pending marks, they belong to previous base character.
            if base_chars and pending_marks:
                # attach pending marks as label to previous char (concatenate if needed)
                prev_label = labels[-1]
                if prev_label == "_":
                    labels[-1] = "".join(pending_marks)
                else:
                    labels[-1] = prev_label + "".join(pending_marks)
                pending_marks = []
            # start new base char
            base_chars.append(ch)
            labels.append("_")  # default label: no diacritic yet

    # after loop: if leftover combining marks (no base char after them)
    if pending_marks:
        if base_chars:
            # attach trailing marks to the last base char (concatenate if needed)
            prev_label = labels[-1]
            if prev_label == "_":
                labels[-1] = "".join(pending_marks)
            else:
                labels[-1] = prev_label + "".join(pending_marks)
            pending_marks = []
        else:
            # No base char at all in the line: log and ignore (leading/trailing stray marks)
            logger.warning(
                "Found trailing/stray combining marks with no base char to attach: %r -> ignored",
                "".join(pending_marks),
            )

    base_text = "".join(base_chars)
    return base_text, labels

def normalize_text(line: str, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Normalize the Arabic text. Options dict supports:
      - 'normalize_hamza': bool (default True) -> convert hamza variants to a canonical form (ا/أ/إ/آ -> ا)
      - 'remove_tatweel': bool (default True) -> remove tatweel U+0640
      - 'lower_latin': bool (default True) -> lower-case Latin fragments
      - 'remove_punctuation': bool (default False) -> optionally strip punctuation (not recommended by default)
    NOTE: normalization deliberately preserves diacritics if present. For training inputs we typically call
    strip_diacritics() separately when we want undiacritized inputs.
    """
    if options is None:
        options = {}
    normalize_hamza = options.get("normalize_hamza", True)
    remove_tatweel = options.get("remove_tatweel", True)
    lower_latin = options.get("lower_latin", True)
    remove_punct = options.get("remove_punctuation", False)
    print(f"Normalization options: {options}")

    s = line
    # normalize to NFKC for stability
    s = unicodedata.normalize("NFKC", s)
    # remove tatweel
    if remove_tatweel:
        s = s.replace("\u0640", "")
        print("Removed tatweel characters.")

    if normalize_hamza:
        s = s.replace("\u0622", "\u0627")  # آ -> ا
        s = s.replace("\u0623", "\u0627")  # أ -> ا
        s = s.replace("\u0625", "\u0627")  # إ -> ا
        # optionally other normalizations can be added

    if lower_latin:
        # lower-case any ASCII/Latin letters
        s = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), s)

    if remove_punct:
        s = re.sub(r"[^\p{L}\p{N}\s]", " ", s)  # keep letters, numbers, whitespace

    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_char_vocab(texts: Iterable[str], min_freq: int = 1) -> Tuple[Dict[str,int], Dict[int,str]]:
    """
    Build char2idx and idx2char from a list/iterable of base_text strings.
    Deterministic ordering: sort by frequency desc, then by unicode codepoint asc.
    Always reserve:
      0 -> PAD_TOKEN
      1 -> UNK_TOKEN
    Returns (char2idx, idx2char)
    """
    counter = Counter()
    for t in texts:
        counter.update(list(t))
    # filter by min_freq
    items = [(ch, freq) for ch, freq in counter.items() if freq >= min_freq]
    # sort by freq desc, then codepoint ascending
    items.sort(key=lambda x: (-x[1], ord(x[0])))
    # build mapping
    char2idx: Dict[str,int] = OrderedDict()
    idx2char: Dict[int,str] = {}
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
    """
    Build label2idx and idx2label. Always include '_' (no diacritic).
    Deterministic ordering: sort labels lexicographically, but keep '_' as index 0.
    """
    s = set()
    for lablist in labels:
        s.update(lablist)
    if "_" not in s:
        s.add("_")
    # deterministic ordering (but keep '_' first)
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
    """
    PyTorch Dataset yielding dictionary items:
      {
        'chars': LongTensor(seq_len),
        'labels': LongTensor(seq_len),
        'raw': original base_text (no diacritics),
        'label_strs': list[str] (labels per character)
      }
    make_dataset() below constructs this dataset.
    """
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
    """
    Pads 'chars' and 'labels'. For labels we use PAD_LABEL (-100) for padded positions
    so that PyTorch losses (like CrossEntropyLoss with ignore_index=-100) ignore them.
    Returns:
      chars: LongTensor(batch, max_len)
      labels: LongTensor(batch, max_len) with PAD_LABEL in padded positions
      mask: BoolTensor(batch, max_len) True for real tokens
      raws: list[str]
      label_strs: list[list[str]]
    """
    chars = [item["chars"] for item in batch]
    labels = [item["labels"] for item in batch]
    raws = [item["raw"] for item in batch]
    label_strs = [item["label_strs"] for item in batch]

    lengths = [c.size(0) for c in chars]
    max_len = max(lengths)
    padded_chars = pad_sequence(chars, batch_first=True, padding_value=pad_idx)  # (B, L)
    # labels: pad with PAD_LABEL
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
                 strip_input_diacritics: bool = True, normalization_options: Optional[Dict]={"normalize_hamza": True, "remove_tatweel": True, "lower_latin": True, "remove_punctuation": True}) -> DiacritizationDataset:
    """
    Read file at file_path (one sentence per line; expected diacritized for train/dev),
    produce processed entries list and return Dataset.
    
    Each processed entry (dictionary) contains:
      - 'raw': base_text (no diacritics after normalization)
      - 'base': same as raw (for clarity)
      - 'labels': list[str] (label string per char)
      - 'char_ids': list[int] mapped by char2idx (UNK if missing)
      - 'label_ids': list[int] mapped by label2idx
    """
    entries = []
    detect_encoding(file_path)  # will raise if not utf-8
    with open(file_path, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            line = clean_line(line)
            if len(line.strip()) == 0:
                continue
            base_text, labels = extract_labels(line)
            if len(base_text) == 0:
                logger.warning("Line %d -> base_text empty after extract; skipping", i)
                continue
            # optional normalization
            if normalization_options:
                base_text = normalize_text(base_text, normalization_options)
            # optionally strip diacritics from input (we keep labels separate)
            if strip_input_diacritics:
                base_text = strip_diacritics(base_text)
            # map chars -> ids
            char_ids = [char2idx.get(ch, char2idx.get(UNK_TOKEN)) for ch in base_text]
            # map labels -> ids (ensure lengths equal)
            if len(labels) != len(base_text):
                # attempt to realign: easiest fallback is to create '_' labels for mismatch and log
                logger.warning("Label/base length mismatch on line %d (%d vs %d). Adjusting labels to '_' for safety.",
                               i, len(labels), len(base_text))
                labels = ["_"] * len(base_text)
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
    """
    Example pipeline:
      - read input_path
      - build char vocab & label map
      - save char2idx.json & label2idx.json
      - save small processed JSONL (one JSON per line)
    """
    os.makedirs(out_dir, exist_ok=True)
    # quick pass: collect base_texts and labels
    detect_encoding(input_path)
    base_texts = []
    label_lists = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            line = clean_line(line)
            if not line:
                continue
            base, labs = extract_labels(line)
            if not base:
                continue
            base_texts.append(strip_diacritics(base))
            label_lists.append(labs)
    char2idx, idx2char = build_char_vocab(base_texts, min_freq=min_freq)
    label2idx, idx2label = build_label_map(label_lists)
    save_json(char2idx, os.path.join(out_dir, "char2idx.json"))
    save_json(idx2char, os.path.join(out_dir, "idx2char.json"))
    save_json(label2idx, os.path.join(out_dir, "label2idx.json"))
    save_json(idx2label, os.path.join(out_dir, "idx2label.json"))
    # write small processed jsonl
    jsonl_path = os.path.join(out_dir, "processed_sample.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for base, labs in zip(base_texts, label_lists):
            char_ids = [char2idx.get(ch, char2idx.get(UNK_TOKEN)) for ch in base]
            label_ids = [label2idx.get(lab, label2idx.get("_")) for lab in labs] if len(labs) == len(base) else [label2idx.get("_")] * len(base)
            rec = {
                "raw": base,
                "base": base,
                "labels": labs,
                "char_ids": char_ids,
                "label_ids": label_ids
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved vocab and sample processed JSONL to %s", out_dir)
    return os.path.join(out_dir, "char2idx.json"), os.path.join(out_dir, "label2idx.json")


if __name__ == "__main__":
    example_usage()