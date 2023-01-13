import unittest

import torch

from data import ptb_loaders


class TestData(unittest.TestCase):
    def test_ptb(self):
        n_seq = 3
        n_batch = 16
        n_vocab = 128
        train, val = ptb_loaders(n_seq, n_batch)
        tx, ty = next(iter(train))
        vx, vy = next(iter(train))

        self.assertEqual(tx.shape, (n_batch, n_seq, n_vocab))
        self.assertEqual(ty.shape, (n_batch, n_seq))
        self.assertEqual(tx.shape, vx.shape)
        self.assertEqual(ty.shape, vy.shape)
        self.assertTrue(tx.dtype is torch.float32)
        self.assertTrue(ty.dtype is torch.int64)