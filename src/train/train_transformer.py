# A minimal train script that uses HF Trainer for token classification.
# This script tokenizes sentences as characters (so each character aligns to a token)
import os
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from ..config import cfg
from ..utils.checkpoints import save_checkpoint

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def build_char_label_map():
    labels = ['_', '\u064b','\u064c','\u064d','\u064e','\u064f','\u0650','\u0651','\u0652']
    return {l:i for i,l in enumerate(labels)}

def prepare_examples(lines, tokenizer, label_map):
    examples = {'chars': [], 'labels': []}
    for line in lines:
        # naive char and label extraction similar to dataset
        import unicodedata
        base_chars = []
        labels = []
        for ch in line:
            cat = unicodedata.category(ch)
            if cat.startswith('M'):
                if base_chars:
                    labels[-1] = labels[-1] + ch
            else:
                base_chars.append(ch)
                labels.append('_')
        base_text = ''.join(base_chars)
        examples['chars'].append(list(base_text))  # list of single-character tokens
        examples['labels'].append([label_map.get(l, label_map['_']) for l in labels])
    return examples

def run():
    os.makedirs(cfg.models_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_model_name, use_fast=True)

    label_map = build_char_label_map()
    inv_map = {v:k for k,v in label_map.items()}
    train_lines = read_lines(cfg.train_file)
    dev_lines = read_lines(cfg.dev_file)

    train_examples = prepare_examples(train_lines, tokenizer, label_map)
    dev_examples = prepare_examples(dev_lines, tokenizer, label_map)

    train_ds = Dataset.from_dict(train_examples)
    dev_ds = Dataset.from_dict(dev_examples)

    # tokenization function expects 'chars' as list of characters; set is_split_into_words=True
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['chars'], is_split_into_words=True, padding='max_length', truncation=True, max_length=512)
        labels = []
        for i, lab in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(lab[word_idx])
                else:
                    # Subsequent sub-token of same word (character) - keep label or -100
                    label_ids.append(lab[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = train_ds.map(tokenize_and_align_labels, batched=True, remove_columns=['chars','labels'])
    tokenized_dev = dev_ds.map(tokenize_and_align_labels, batched=True, remove_columns=['chars','labels'])

    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
    model = AutoModelForTokenClassification.from_pretrained(cfg.transformer_model_name, num_labels=len(label_map))

    training_args = TrainingArguments(
        output_dir=cfg.models_dir,
        evaluation_strategy="epoch",
        num_train_epochs=cfg.transformer_epochs,
        per_device_train_batch_size=cfg.transformer_batch_size,
        per_device_eval_batch_size=cfg.transformer_batch_size,
        learning_rate=cfg.transformer_lr,
        save_strategy="epoch",
        logging_dir=cfg.logs_dir,
        fp16=torch.cuda.is_available()
    )

    # simple compute_metrics using DER on the tokenized alignment, but here we return accuracy as example
    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # flatten and compare skipping -100
        true_preds = []
        true_labels = []
        for p, l in zip(preds, labels):
            for pred_id, label_id in zip(p, l):
                if label_id != -100:
                    true_preds.append(pred_id)
                    true_labels.append(label_id)
        acc = (np.array(true_preds) == np.array(true_labels)).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(cfg.models_dir)
    tokenizer.save_pretrained(cfg.models_dir)

if __name__ == "__main__":
    run()
