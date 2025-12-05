# BiLSTM + linear + CRF on top
import torch
import torch.nn as nn
from .crf_layer import CRFLayer

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, lstm_hidden, num_labels, num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(input_size=char_emb_dim, hidden_size=lstm_hidden, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.linear = nn.Linear(2*lstm_hidden, num_labels)
        self.crf = CRFLayer(num_labels)

    def forward(self, chars, labels=None, mask=None):
        emb = self.embedding(chars)  # (batch, seq_len, emb)
        out, _ = self.bilstm(emb)    # (batch, seq_len, 2*hidden)
        emissions = self.linear(out) # (batch, seq_len, num_tags)
        if labels is not None:
            loss = self.crf(emissions, labels, mask)
            return loss
        else:
            # decode
            preds = self.crf.decode(emissions, mask)
            return preds
