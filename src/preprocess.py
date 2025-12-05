# text cleaning and normalization utilities
import re
from pyarabic.araby import normalize_hamza, strip_tashkeel, tokenize
from typing import List

# Example normalizer - adapt to dataset choices
ARABIC_LETTERS = re.compile(r'[\u0600-\u06FF]+')

def clean_line(line: str) -> str:
    """Basic cleaning: trim, remove URLs/emails, normalize whitespace."""
    line = line.strip()
    # remove urls
    line = re.sub(r'http\S+|www\.\S+', '', line)
    # remove emails
    line = re.sub(r'\S+@\S+', '', line)
    # normalize repeated whitespace
    line = re.sub(r'\s+', ' ', line)
    return line

def normalize_text(line: str, remove_diacritics: bool = False) -> str:
    """Normalize hamzas and optionally strip diacritics (tashkeel)."""
    line = normalize_hamza(line)
    if remove_diacritics:
        line = strip_tashkeel(line)
    return line

def is_arabic_word(w: str) -> bool:
    return bool(ARABIC_LETTERS.search(w))
