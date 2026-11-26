"""Tests for the drifting-stream task generator."""
import unittest

import numpy as np

from tasks.drift import generate


class Drift(unittest.TestCase):
    def test_seed_reproducible_and_distinct(self):
        X1, y1, _ = generate(123, 4000)
        X2, y2, _ = generate(123, 4000)
        self.assertEqual(X1.tobytes(), X2.tobytes())
        self.assertEqual(y1.tobytes(), y2.tobytes())

        X3, y3, _ = generate(124, 4000)
        self.assertNotEqual(X1.tobytes(), X3.tobytes())
        self.assertNotEqual(y1.tobytes(), y3.tobytes())

    def test_balance_and_drift(self):
        n, dim, segments = 4000, 16, 4
        X, y, W = generate(7, n, dim, segments)
        block = n // segments

        for s in range(segments):
            share = y[s * block:(s + 1) * block].mean()
            self.assertGreaterEqual(share, 0.4)
            self.assertLessEqual(share, 0.6)

        # Decode the byte-quantized stream back near the original x.
        Xf = X.astype(np.float64) / 255.0 * 8.0 - 4.0
        agree0 = ((Xf[:block] @ W[0] > 0) == y[:block]).mean()
        self.assertGreater(agree0, 0.95)

        last = slice((segments - 1) * block, n)
        agree_last = ((Xf[last] @ W[0] > 0) == y[last]).mean()
        self.assertLess(agree_last, 0.8)


if __name__ == "__main__":
    unittest.main()
