# training loop for BiLSTM+CRF
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..config import cfg
from ..data.dataset import CharSequenceDataset
from ..data.collate import collate_chars
from ..tokenizer import CharTokenizer
from ..models.bilstm_crf import BiLSTMCRF
from ..eval.metrics import DERCalculator
from ..utils.checkpoints import save_checkpoint, load_checkpoint

def build_label_map():
    # Simplified label set: common Arabic diacritics + '_' for none
    labels = ['_', '\u064b','\u064c','\u064d','\u064e','\u064f','\u0650','\u0651','\u0652']
    # _ none, tanween etc. Map to indices
    return {l:i for i,l in enumerate(labels)}

def train():
    os.makedirs(cfg.models_dir, exist_ok=True)
    # build char tokenizer from train file
    with open(cfg.train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    char_tok = CharTokenizer()
    char_tok.fit_on_texts(texts, min_freq=cfg.char_vocab_min_freq)

    label_map = build_label_map()
    train_ds = CharSequenceDataset(cfg.train_file, char_tok, label_map)
    dev_ds = CharSequenceDataset(cfg.dev_file, char_tok, label_map)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_chars(b, pad_value=char_tok.char2idx[CharTokenizer.PAD]))
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: collate_chars(b, pad_value=char_tok.char2idx[CharTokenizer.PAD]))

    model = BiLSTMCRF(vocab_size=len(char_tok.char2idx), char_emb_dim=cfg.char_emb_dim, lstm_hidden=cfg.lstm_hidden, num_labels=len(label_map), num_layers=cfg.lstm_layers, dropout=cfg.bilstm_dropout, pad_idx=char_tok.char2idx[CharTokenizer.PAD])
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_dev = 1.0
    der_calc = DERCalculator()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            chars = batch['chars'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            optimizer.zero_grad()
            loss = model(chars, labels=labels, mask=mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # eval
        model.eval()
        all_preds = []
        all_refs = []
        with torch.no_grad():
            for batch in dev_loader:
                chars = batch['chars'].to(device)
                labels = batch['labels'].to(device)
                mask = batch['mask'].to(device)
                preds = model(chars, labels=None, mask=mask)
                # preds: list of list of ints
                for p, lab, m, raw, labstr in zip(preds, labels.cpu().numpy(), mask.cpu().numpy(), batch['raws'], batch['label_strs']):
                    # truncate using mask len
                    seq_len = m.sum()
                    pred = p[:seq_len]
                    ref = lab[:seq_len].tolist()
                    # convert ref ints to diacritic str for DER calc using reverse mapping
                    # create reverse map
                    rev_label = {v:k for k,v in label_map.items()}
                    pred_labels = [rev_label.get(i, '_') for i in pred]
                    ref_labels = [rev_label.get(i, '_') for i in ref]
                    all_preds.append((raw, pred_labels))
                    all_refs.append((raw, ref_labels))
        der = der_calc.compute(all_refs, all_preds)
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f} dev_DER={der:.4f}")
        if der < best_dev:
            best_dev = der
            save_path = os.path.join(cfg.models_dir, 'best_bilstm_crf.pt')
            save_checkpoint({'model_state': model.state_dict(), 'char2idx': char_tok.char2idx, 'label_map': label_map}, save_path)
            print("Saved best model", save_path)

if __name__ == "__main__":
    train()
