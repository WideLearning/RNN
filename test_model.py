import unittest

import torch
from torch import nn
from rnn import RNN
from transformer import causal_attention_mask


class TestRNN(unittest.TestCase):
    def test_shapes(self):
        batch_size, h_size, x_size, y_size = 10, 4, 2, 3
        net = RNN(x_size=x_size, h_size=h_size, y_size=y_size, config={})
        state = net.init_state(std=1, batch_size=batch_size)
        data = torch.randn((batch_size, x_size))

        result, state = net(data, state)

        self.assertEqual(result.shape, (batch_size, y_size))
        self.assertEqual(state.shape, (batch_size, h_size))

    def test_growth(self):
        n_it, batch_size, h_size, x_size, y_size = 100, 10, 4, 2, 3
        net = RNN(h_size=h_size, x_size=x_size, y_size=y_size, config={})
        state = torch.randn((batch_size, h_size))
        loss = 0

        for i in range(n_it):
            data = torch.randn((batch_size, x_size))
            result, state = net(data, state)
            self.assertLess(result.abs().mean(), 10)
            loss += result.sum(dim=0)[0]

        loss.backward()
        for name, par in net.named_parameters():
            self.assertLess(par.grad.abs().mean() / n_it, 10)


class TestTransformer(unittest.TestCase):
    def test_attention(self):
        n, pos, d = 7, 2, 16
        attn = nn.MultiheadAttention(embed_dim=d, num_heads=4)
        q = torch.randn((n, d), requires_grad=True)
        k = torch.randn((n, d), requires_grad=True)
        v = torch.randn((n, d), requires_grad=True)
        m = causal_attention_mask(n)
        
        y, w = attn(q, k, v, attn_mask=m)
        y[pos].sum().backward()

        self.assertTrue(m.dtype is torch.bool)
        self.assertEqual(y.shape, (n, d))
        self.assertEqual(w.shape, (n, n))
        qg, kg, vg = q.grad.sum(dim=1), k.grad.sum(dim=1), v.grad.sum(dim=1)
        for i in range(n):
            self.assertEqual(qg[i] != 0, i == pos)
            self.assertEqual(kg[i] != 0, i <= pos)
            self.assertEqual(vg[i] != 0, i <= pos)
