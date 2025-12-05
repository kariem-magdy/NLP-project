import os
from src.config import cfg
from src.preprocess import (
    clean_line, 
    extract_labels, 
    build_char_vocab, 
    build_label_map, 
    save_json
)

def build_vocabs():
    # 1. Define paths
    train_path = cfg.train_file  # Usually 'data/train.txt'
    processed_dir = os.path.join(cfg.outputs_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"[INFO] Reading training data from {train_path}...")
    
    # 2. Collect all characters and labels from training data
    all_chars = []
    all_labels_lists = []
    
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_line(line.rstrip("\n"))
            if not line: continue
            
            # Extract text (without diacritics) and labels (the diacritics)
            base_text, labels = extract_labels(line)
            
            all_chars.append(base_text)
            all_labels_lists.append(labels)
            
    print("[INFO] Building vocabularies...")
    
    # 3. Build the mappings
    char2idx, idx2char = build_char_vocab(all_chars)
    label2idx, idx2label = build_label_map(all_labels_lists)
    
    print(f"[INFO] Found {len(char2idx)} unique characters and {len(label2idx)} unique diacritic labels.")
    
    # 4. Save to JSON files
    save_json(char2idx, os.path.join(processed_dir, "char2idx.json"))
    save_json(idx2char, os.path.join(processed_dir, "idx2char.json"))
    save_json(label2idx, os.path.join(processed_dir, "label2idx.json"))
    save_json(idx2label, os.path.join(processed_dir, "idx2label.json"))
    
    print(f"[SUCCESS] Vocab files saved to {processed_dir}")

if __name__ == "__main__":
    build_vocabs()