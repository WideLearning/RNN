import unittest
from data import load_ptb, codes_to_text, codes_to_onehot

class TestData(unittest.TestCase):
    def test_dataset(self):
        dataset = load_ptb()
        piece = dataset[800:810]

        self.assertEqual(codes_to_text(piece), "earchers s")
        self.assertEqual(codes_to_onehot(piece).shape, (10, 128))