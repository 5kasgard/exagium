"""Offline tests for scale.py (no jail, no real timing). Run: cd loop && python3 -m unittest test_scale"""
import os, sys, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop as L  # noqa: E402
import scale  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MLP16 = os.path.join(HERE, "candidates", "mlp16")


class Pool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.y = scale.pool(65536, 1)

    def test_distinct_in_range_and_labelled(self):
        self.assertEqual(len(self.p), 65536)
        self.assertEqual(len({tuple(r) for r in self.p}), 65536)
        self.assertTrue(all(len(r) == 64 and all(0 <= v <= 16 for v in r) for r in self.p))
        self.assertEqual(len(self.y), 65536)
        self.assertEqual(set(self.y), set(range(10)))

    def test_never_uses_dev_or_test_rows(self):
        held = {tuple(r[:64]) for r in L.split("dev") + L.split("test")}
        train = L.load(L.TASK["train"])
        train = {tuple(r[:64]) for r in train[:len(train) * 3 // 4]}
        # a held-out image may equal a training image by chance; anything else in the pool is a leak
        self.assertEqual([r for r in self.p if tuple(r) in held and tuple(r) not in train], [])

    def test_labels_follow_their_images(self):
        train = L.load(L.TASK["train"])
        label = {tuple(r[:64]): r[64] for r in train[:len(train) * 3 // 4]}
        hits = [(tuple(im), y) for im, y in zip(self.p, self.y) if tuple(im) in label]
        self.assertGreater(len(hits), 100)
        self.assertTrue(all(label[im] == y for im, y in hits))

    def test_seeded(self):
        # measure() takes prefixes of one pool; the pool for a given (n, seed) must be the same every time
        self.assertTrue(scale.pool(65536, 1)[0] == self.p)
        self.assertFalse(scale.pool(65536, 2)[0] == self.p)

    def test_starts_from_every_original_and_nudges_stay_close(self):
        train = L.load(L.TASK["train"])
        orig = {tuple(r[:64]) for r in train[:len(train) * 3 // 4]}
        pool = {tuple(r) for r in self.p}
        self.assertTrue(orig <= pool)
        import random
        rng = random.Random(0)
        row = list(range(17)) * 3 + [0] * 13
        n = scale.nudged(row, rng)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(row, n)), scale.NUDGES)
        self.assertTrue(all(0 <= v <= 16 for v in n))


class Summary(unittest.TestCase):
    SIZES = [0, 956, 4096, 16384, 65536]

    def rounds(self, a, b, shifts=(0, 0, 0)):
        return [{n: a + s + b * n for n in self.SIZES} for s in shifts]

    def test_fixed_marginal_segments_and_task_formula(self):
        per, _ = scale.summarise({"x": self.rounds(1e6, 30.0)}, self.SIZES)
        x = per["x"]
        self.assertAlmostEqual(x["fixed_ns_t0"][0], 1e6)
        self.assertAlmostEqual(x["marginal_ns_per_image_large_n"][0], 30.0)
        self.assertEqual(x["large_n_segment"], [16384, 65536])
        self.assertEqual([(g["from"], g["to"]) for g in x["segments"]], list(zip(self.SIZES, self.SIZES[1:])))
        self.assertTrue(all(abs(g["ns_per_image"][0] - 30.0) < 1e-9 for g in x["segments"]))
        self.assertAlmostEqual(x["n956_pool_minus_t0_ns_per_image"][0], 30.0)

    def test_marginal_uses_the_two_largest_sizes_only(self):
        r = [{0: 1e6, 956: 2e6, 4096: 3e6, 16384: 4e6, 65536: 4e6 + 49152 * 40.0}]
        per, _ = scale.summarise({"x": r}, self.SIZES)
        self.assertAlmostEqual(per["x"]["marginal_ns_per_image_large_n"][0], 40.0)

    def test_pairs_are_paired_by_round(self):
        shifts = (0, 5e5, -3e5)  # a per-round shift shared by both candidates cancels
        _, (p,) = scale.summarise({"x": self.rounds(1e6, 30.0, shifts), "y": self.rounds(1e6, 50.0, shifts)}, self.SIZES)
        self.assertEqual((p["a"], p["b"]), ("x", "y"))
        self.assertAlmostEqual(p["marginal_diff_ns"], -20.0)
        self.assertAlmostEqual(p["marginal_diff_se"], 0.0)
        self.assertAlmostEqual(p["n956_diff_ns"], -20.0)
        self.assertAlmostEqual(p["n956_diff_se"], 0.0)


class Accuracy(unittest.TestCase):
    def test_honest_candidate_passes(self):
        y = [i % 10 for i in range(5000)]
        pred = [v if i % 10 else (v + 1) % 10 for i, v in enumerate(y)]  # 90% everywhere
        self.assertIsNone(scale.accuracy_check(pred, y, [0, 956, 2000, 5000])["problem"])

    def test_less_work_at_large_n_is_a_problem(self):
        y = [i % 10 for i in range(5000)]
        pred = y[:1000] + [7] * 4000  # the reviewer's cheat: real answers early, a constant later
        self.assertIsNotNone(scale.accuracy_check(pred, y, [0, 956, 2000, 5000])["problem"])


class AccuracyMiddle(unittest.TestCase):
    def test_cheating_only_in_a_middle_segment_is_caught(self):
        # free outside review (gpt-oss-120b, 2026-11-18): a cheat confined to one middle segment passed the old check
        y = [i % 10 for i in range(20000)]
        pred = y[:4096] + [7] * (16384 - 4096) + y[16384:]
        rec = scale.accuracy_check(pred, y, [0, 956, 4096, 16384, 20000])
        self.assertIn("4096-16384", rec["problem"])


class Measure(unittest.TestCase):
    """measure() with the jail, compiler and timer faked: the checks must fire."""

    def run_measure(self, behaviour, built=None):
        calls = {"n": 0}
        built = [] if built is None else built

        def fake_timed(c, rows, timer, cpu):
            calls["n"] += 1
            return behaviour(rows, calls["n"]), 1_000_000 + 50 * len(rows)

        with mock.patch.object(L, "build", lambda src, out: built.append(("host", src))), \
             mock.patch.object(scale.J, "compile_candidate", lambda src, out: built.append(("jail", src))), \
             mock.patch.object(L, "timed", fake_timed), \
             mock.patch.dict(L.JAIL, {"mode": "confine"}), \
             mock.patch.object(L, "versions", lambda: {}):
            return scale.measure([MLP16], [0, 956, 4096], 2, 1)

    def test_candidates_compile_in_the_jail_and_only_the_timer_on_the_host(self):
        """Q56b: candidate C is untrusted, so it never reaches the host gcc; the trusted timer still does."""
        built = []
        self.run_measure(lambda rows, k: self.labels_for(rows), built)
        self.assertEqual([w for w, s in built if w == "host"], ["host"])
        self.assertEqual([s for w, s in built if w == "host"], [L.TIMER])
        jailed = [s for w, s in built if w == "jail"]
        self.assertEqual(len(jailed), 1)
        self.assertTrue(jailed[0].endswith(os.path.join("mlp16", "infer.c")))

    def labels_for(self, rows):
        imgs, labels = scale.pool(4096, 1)
        index = {tuple(r): y for r, y in zip(imgs, labels)}
        return [index[tuple(r)] for r in rows]

    def test_honest_candidate_has_no_problems(self):
        s = self.run_measure(lambda rows, k: self.labels_for(rows))
        self.assertEqual(s["problems"], [])
        self.assertAlmostEqual(s["candidates"]["mlp16"]["marginal_ns_per_image_large_n"][0], 50.0)
        self.assertEqual(len(s["raw_ns"]["mlp16"]), 2)

    def test_nondeterministic_predictions_are_a_problem(self):
        s = self.run_measure(lambda rows, k: [(y + (k > 1)) % 10 for y in self.labels_for(rows)])
        self.assertTrue(any("differ from the same images" in p for p in s["problems"]))

    def test_less_work_at_large_n_is_a_problem(self):
        s = self.run_measure(lambda rows, k: [y if i < 1000 else 7 for i, y in enumerate(self.labels_for(rows))])
        self.assertTrue(any("less work at larger N" in p for p in s["problems"]))

    def test_refuses_without_a_jail(self):
        with mock.patch.dict(L.JAIL, {"mode": None}):
            with self.assertRaises(L.Failed):
                scale.measure([MLP16], [0, 956, 4096], 2, 1)


if __name__ == "__main__":
    unittest.main()
