import unittest

import torch

from data import ptb_loaders


class TestData(unittest.TestCase):
    def test_ptb(self):
        seq_len = 3
        batch_size = 16
        vocab_size = 128
        train, val = ptb_loaders(seq_len, batch_size)
        tx, ty = next(iter(train))
        vx, vy = next(iter(train))

        self.assertEqual(tx.shape, (batch_size, seq_len, vocab_size))
        self.assertEqual(ty.shape, (batch_size, seq_len))
        self.assertEqual(tx.shape, vx.shape)
        self.assertEqual(ty.shape, vy.shape)
        self.assertTrue(tx.dtype is torch.float32)
        self.assertTrue(ty.dtype is torch.int64)