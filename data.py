import torch
import torch.nn.functional as F

def load_ptb():
    return torch.load("ptb.p").to(dtype=int)

def codes_to_text(arr):
    return "".join(chr(c) for c in arr)

def codes_to_onehot(arr):
    return F.one_hot(arr, num_classes=128)