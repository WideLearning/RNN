import unittest

import torch

from model import RNN


class TestModel(unittest.TestCase):
    def test_shapes(self):
        batch_size, h_size, x_size, y_size = 10, 4, 2, 3
        net = RNN(x_size=x_size, h_size=h_size, y_size=y_size)
        state = net.init_state(std=1, batch_size=batch_size)
        data = torch.randn((batch_size, x_size))

        result, state = net(data, state)

        self.assertEqual(result.shape, (batch_size, y_size))
        self.assertEqual(state.shape, (batch_size, h_size))

    def test_growth(self):
        n_it, batch_size, h_size, x_size, y_size = 100, 10, 4, 2, 3
        net = RNN(h_size=h_size, x_size=x_size, y_size=y_size)
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
