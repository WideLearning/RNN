import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def codes_to_text(arr):
    return "".join(chr(c) for c in arr)


class SequenceDataset(Dataset):
    def __init__(self, data, n_seq, n_vocab):
        self.data = data
        self.n_seq = n_seq
        self.n_vocab = n_vocab

    def __len__(self):
        return len(self.data) - self.n_seq

    def __getitem__(self, idx):
        x = F.one_hot(self.data[idx : idx + self.n_seq], num_classes=self.n_vocab).float()
        idx += 1
        y = self.data[idx : idx + self.n_seq].long()
        return x, y


def ptb_loaders(n_seq, n_batch, train_ratio=0.9):
    data = torch.load("ptb.p").to(dtype=int)
    n_train = int(train_ratio * len(data))
    n_vocab = 128
    ds_train = SequenceDataset(data[:n_train], n_seq=n_seq, n_vocab=n_vocab)
    ds_val = SequenceDataset(data[:n_train], n_seq=n_seq, n_vocab=n_vocab)
    dl_train = DataLoader(ds_train, n_batch, shuffle=True)
    dl_val = DataLoader(ds_val, n_batch, shuffle=False)
    return dl_train, dl_val
