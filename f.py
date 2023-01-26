import matplotlib.pyplot as plt
import numpy as np

seq_len = 10

def loss(k0, k1, w0, w1):
    answer = k0 % 2
    prediction = (k0 * w0 + k1 * w1) % 2
    return abs(answer - prediction)

def only_last(w0, w1):
    s = 0
    c = 0
    for k0 in range(seq_len + 1):
        k1 = seq_len - k0
        s += loss(k0, k1, w0, w1)
        c += 1
    return s / c

def full(w0, w1):
    s = 0
    c = 0
    for k0 in range(seq_len + 1):
        for k1 in range(seq_len + 1 - k0):
            s += loss(k0, k1, w0, w1)
            c += 1
    return s / c

sp = np.linspace(-2, 2, 1000)
x, y = np.meshgrid(sp, sp)
z = x
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        z[i, j] = only_last(x[i, j], y[i, j])
print(x.shape, y.shape, z.shape)
plt.matshow(z)
plt.colorbar()
plt.show()