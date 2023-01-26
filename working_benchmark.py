import functools
import operator
import random


def foldl(func, acc, xs):
    return functools.reduce(func, xs, acc)


random.seed(1)


def trace_xor(a, b):
    """
    shows the intermediate steps of
    xor function on
    a sequence
    """
    result = operator.xor(a, b)
    print(f"{a} XOR {b} = {result}")
    return result


import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
# Data
TRAINING_SIZE = 100000
VALIDATION_SIZE = 10000
BIT_LEN = 50
VARIABLE_LEN = True

# Model Parameters
INPUT_SIZE = 2
HIDDEN_SIZE = 2
NUM_LAYERS = 1

# Training Parameters
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.01  # DEFAULT ADAM 0.001

THRESHOLD = 0.0001


class XOR(data.Dataset):
    """GENERATE XOR DATA"""

    def __init__(self, sample_size=VALIDATION_SIZE, bit_len=BIT_LEN, variable=False):
        self.bit_len = bit_len
        self.sample_size = sample_size
        self.variable = VARIABLE_LEN
        self.features, self.labels = self.generate_data(sample_size, bit_len)

    def __getitem__(self, index):
        return self.features[index, :], self.labels[index]

    def __len__(self):
        return len(self.features)

    def generate_data(self, sample_size, seq_length=BIT_LEN):

        bits = torch.randint(2, size=(sample_size, seq_length, 1)).float()
        if self.variable:
            # we generate random integers and pad the bits with zeros
            # to mimic variable bit string lengths
            # padding with zeros as they do not provide information
            # TODO: vectorize instead of loop?
            pad = torch.randint(seq_length, size=(sample_size,))
            for idx, p in enumerate(pad):
                bits[idx, p:] = 0.0

        bitsum = bits.cumsum(axis=1)
        # if bitsum[i] odd: -> True
        # else: False
        parity = (bitsum % 2 != 0).long().squeeze(2)
        parity[:, :-1] = -100
        return F.one_hot(bits.long().squeeze(2), num_classes=2).float(), parity

class XORLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(XORLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x, lengths=True):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out_lstm, _ = self.lstm(x, (h0, c0))
        out = self.fc(out_lstm)

        predictions = self.activation(out)
        return predictions


model = XORLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(
    XOR(TRAINING_SIZE, BIT_LEN, 5), batch_size=1
)
x, y = next(iter(train_loader))
# print(x.shape, x.dtype)
# print(y.shape, y.dtype)
# print(x)
# print(y)
# exit(0)

# train
def train():
    model.train()
    train_loader = DataLoader(
        XOR(TRAINING_SIZE, BIT_LEN, VARIABLE_LEN), batch_size=BATCH_SIZE
    )
    total_step = len(train_loader)

    print("Training...\n")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features).permute(0, 2, 1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (
                (outputs.argmax(dim=1) == (labels > 0.5)).float().mean()
            )

            if (step + 1) % 250 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}".format(
                        epoch, EPOCHS, step + 1, total_step, loss.item(), accuracy
                    )
                )
                print("-" * 60)
                if abs(accuracy - 1.0) < THRESHOLD:
                    print("EARLY STOPPING")
                    return

            if step + 1 == total_step:
                valid_accuracy = validate(model)
                print("validation accuracy: {:.4f}".format(valid_accuracy))
                print("-" * 60)
                if abs(valid_accuracy - 1.0) < THRESHOLD:
                    print("EARLY STOPPING")
                    return


def validate(model):
    valid_loader = DataLoader(
        XOR(VALIDATION_SIZE, BIT_LEN, VARIABLE_LEN), batch_size=BATCH_SIZE
    )
    model.eval()
    correct = 0.0
    total = 0.0
    for features, labels in valid_loader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(features)
            total += labels.size(0) * labels.size(1)
            correct += ((outputs > 0.5) == (labels > 0.5)).sum().item()
    return correct / total

train()