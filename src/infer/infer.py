# simple inference script for the BiLSTM+CRF model.
import argparse
import torch
from ..utils.checkpoints import load_checkpoint
from ..models.bilstm_crf import BiLSTMCRF
from ..tokenizer import CharTokenizer
from ..config import cfg
from ..eval.metrics import DERCalculator

def infer_single_sentence(sentence: str, model_path: str):
    ckpt = load_checkpoint(model_path, device=cfg.device)
    char2idx = ckpt['char2idx']
    label_map = ckpt['label_map']
    rev_label = {v:k for k,v in label_map.items()}
    # load tokenizer
    tok = CharTokenizer()
    tok.char2idx = char2idx
    tok.idx2char = {v:k for k,v in char2idx.items()}
    model = BiLSTMCRF(vocab_size=len(tok.char2idx), char_emb_dim=cfg.char_emb_dim, lstm_hidden=cfg.lstm_hidden, num_labels=len(label_map), num_layers=cfg.lstm_layers, dropout=cfg.bilstm_dropout, pad_idx=tok.char2idx[CharTokenizer.PAD])
    model.load_state_dict(ckpt['model_state'])
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()
    # prepare input
    char_idxs = torch.tensor([tok.text_to_sequence(sentence)], dtype=torch.long).to(device)
    mask = (char_idxs != tok.char2idx[CharTokenizer.PAD]).bool()
    preds = model(char_idxs, labels=None, mask=mask)[0]  # first sample
    pred_labels = [rev_label.get(i, '_') for i in preds]
    # reconstruct diacritized string: naive approach: append diacritic char after base char if non '_' 
    out = []
    for ch, lab in zip(sentence, pred_labels):
        out.append(ch)
        if lab != '_' and lab != '':
            out.append(lab)
    return ''.join(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sentence', required=True)
    args = parser.parse_args()
    print(infer_single_sentence(args.sentence, args.model))
