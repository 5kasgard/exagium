"""Offline tests for scale.py (no jail, no timing). Run: cd loop && python3 -m unittest test_scale"""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop as L  # noqa: E402
import scale  # noqa: E402


class Pool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = scale.pool(65536, 1)

    def test_distinct_and_in_range(self):
        self.assertEqual(len(self.p), 65536)
        self.assertEqual(len({tuple(r) for r in self.p}), 65536)
        self.assertTrue(all(len(r) == 64 and all(0 <= v <= 16 for v in r) for r in self.p))

    def test_never_uses_dev_or_test_rows(self):
        held = {tuple(r[:64]) for r in L.split("dev") + L.split("test")}
        train = L.load(L.TASK["train"])
        train = {tuple(r[:64]) for r in train[:len(train) * 3 // 4]}
        # an unshifted held-out image may equal a training image by chance; anything else in the pool is a leak
        leaks = [r for r in self.p if tuple(r) in held and tuple(r) not in train]
        self.assertEqual(leaks, [])

    def test_seeded(self):
        self.assertEqual(scale.pool(100, 1), self.p[:100])
        self.assertNotEqual(scale.pool(100, 2), self.p[:100])

    def test_too_many_is_refused(self):
        with self.assertRaises(L.Failed):
            scale.pool(10 ** 6, 1)

    def test_shift(self):
        row = list(range(64))
        self.assertEqual(scale.shifted(row, 0, 0), tuple(row))
        s = scale.shifted(row, 1, 0)
        self.assertEqual(s[0], 0)
        self.assertEqual(s[1], row[0])


class Fit(unittest.TestCase):
    def test_ols_exact(self):
        self.assertEqual(scale.ols([0, 10, 20], [5, 25, 45]), (5.0, 2.0))

    def test_ols_needs_two_sizes(self):
        with self.assertRaises(ValueError):
            scale.ols([3, 3], [1, 2])

    def test_summary_recovers_fixed_and_per_image_and_pairs(self):
        sizes = [0, 956, 1024, 1025, 4096]
        def rounds(a, b, jitter):
            return [{n: a + b * n + (j if n == 1025 else 0) for n in sizes} for j in jitter]
        ts = {"x": rounds(1e6, 30.0, [0, 0, 0]), "y": rounds(1e6, 50.0, [0, 0, 0])}
        per, pairs = scale.summarise(ts, sizes)
        self.assertAlmostEqual(per["x"]["ns_per_image"][0], 30.0)
        self.assertAlmostEqual(per["x"]["fixed_ns_per_run"][0], 1e6)
        self.assertAlmostEqual(per["x"]["step_1025_minus_1024_ns"][0], 30.0)
        self.assertAlmostEqual(per["x"]["task_meter_marginal_ns_per_image"][0], 30.0)
        (p,) = pairs
        self.assertEqual((p["a"], p["b"]), ("x", "y"))
        self.assertAlmostEqual(p["ns_per_image_diff"], -20.0)
        self.assertAlmostEqual(p["ns_per_image_diff_se"], 0.0)

    def test_pairs_are_paired_by_round(self):
        # a shared per-round shift cancels in the paired difference
        sizes = [0, 1000]
        ts = {"x": [{0: 1e6 + s, 1000: 1e6 + s + 30000 + 1000 * s / 1000} for s in (0, 5000, -3000)],
              "y": [{0: 1e6 + s, 1000: 1e6 + s + 50000 + 1000 * s / 1000} for s in (0, 5000, -3000)]}
        _, (p,) = scale.summarise(ts, sizes)
        self.assertAlmostEqual(p["ns_per_image_diff"], -20.0)
        self.assertAlmostEqual(p["ns_per_image_diff_se"], 0.0)


if __name__ == "__main__":
    unittest.main()
