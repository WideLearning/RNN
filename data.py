import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def codes_to_text(arr):
    return "".join(chr(c) for c in arr)


class LMDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = F.one_hot(
            self.data[idx : idx + self.seq_len], num_classes=self.vocab_size
        ).float()
        idx += 1
        y = self.data[idx : idx + self.seq_len].long()
        return x, y


def ptb_loaders(seq_len, batch_size, train_ratio=0.9):
    data = torch.load("ptb.p").to(dtype=int)
    n_train = int(train_ratio * len(data))
    vocab_size = 128
    ds_train = LMDataset(data[:n_train], seq_len=seq_len, vocab_size=vocab_size)
    ds_val = LMDataset(data[:n_train], seq_len=seq_len, vocab_size=vocab_size)
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)
    return dl_train, dl_val


class Seq2SeqDataset(Dataset):
    def __init__(self, x, y, vocab_size):
        assert x.dim() == 2  # number of points, sequence length
        assert x.shape == y.shape
        self.x = x.to(dtype=int)
        self.y = y.to(dtype=int)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = F.one_hot(self.x[idx], num_classes=self.vocab_size).float()
        y = self.y[idx].long()
        return x, y


def xor_loaders(seq_count, seq_len, batch_size, train_ratio=0.9, allow_intermediate=False):
    xs, ys = [], []
    NO_VALUE = -100  # NLLLoss ignores y=-100 by default
    for _ in range(seq_count):
        x = torch.randint(high=2, size=(seq_len,))
        y = torch.full(size=(seq_len,), fill_value=NO_VALUE)
        if allow_intermediate:
            for i in range(seq_len):
                y[i] = x[:i].sum() % 2
        else:
            y[-1] = x.sum() % 2
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    Y = torch.stack(ys)
    n_train = int(train_ratio * seq_count)
    ds_train = Seq2SeqDataset(X[:n_train], Y[:n_train], vocab_size=2)
    ds_val = Seq2SeqDataset(X[n_train:], Y[n_train:], vocab_size=2)
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)
    return dl_train, dl_val
