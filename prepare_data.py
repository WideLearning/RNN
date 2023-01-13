from datasets import load_dataset
import torch

dataset = load_dataset("ptb_text_only")
codes = []

for i in range(dataset['train'].num_rows):
    for c in dataset['train'][i]['sentence']:
        codes.append(ord(c))

codes = torch.tensor(codes, dtype=torch.int8)
print(codes.shape)
torch.save(codes, "ptb.p")