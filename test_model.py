import unittest

import torch

from model import RNN


class TestModel(unittest.TestCase):
    def test_shapes(self):
        n_batch, n_h, n_x, n_y = 10, 4, 2, 3
        net = RNN(n_x=n_x, n_h=n_h, n_y=n_y)
        state = net.init_state(std=1, n_batch=n_batch)
        data = torch.randn((n_batch, n_x))

        result, state = net(data, state)

        self.assertEqual(result.shape, (n_batch, n_y))
        self.assertEqual(state.shape, (n_batch, n_h))

    def test_growth(self):
        n_it, n_batch, n_h, n_x, n_y = 100, 10, 4, 2, 3
        net = RNN(n_h=n_h, n_x=n_x, n_y=n_y)
        state = torch.randn((n_batch, n_h))
        loss = 0

        for i in range(n_it):
            data = torch.randn((n_batch, n_x))
            result, state = net(data, state)
            self.assertLess(result.abs().mean(), 10)
            loss += result.sum(dim=0)[0]

        loss.backward()
        for name, par in net.named_parameters():
            self.assertLess(par.grad.abs().mean() / n_it, 10)
