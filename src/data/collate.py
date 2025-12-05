# collate functions for DataLoader
import torch
from typing import List

def collate_chars(batch: List[dict], pad_value: int = 0):
    # pad chars and labels to max length in batch
    max_len = max(x['chars'].size(0) for x in batch)
    batch_size = len(batch)
    chars = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 for ignore index
    masks = torch.zeros((batch_size, max_len), dtype=torch.bool)
    raws = []
    label_strs = []
    for i, item in enumerate(batch):
        l = item['chars'].size(0)
        chars[i, :l] = item['chars']
        labels[i, :l] = item['labels']
        masks[i, :l] = 1
        raws.append(item['raw'])
        label_strs.append(item['label_strs'])
    return {'chars': chars, 'labels': labels, 'mask': masks, 'raws': raws, 'label_strs': label_strs}
