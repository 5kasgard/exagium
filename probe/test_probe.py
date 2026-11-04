"""Tests for forge-probe. Stdlib unittest:  python3 -m unittest -v test_probe"""
import json, math, os, random, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def put(path, text):
    with open(path, "w") as f:
        f.write(text)


def synth_curve(levels, sizes, rel_noise, seed=1):
    """levels: [(capacity_bytes, ns)], last is unbounded. Returns medians and SEs."""
    r = random.Random(seed)
    med, se = [], []
    for s in sizes:
        ns = next(v for cap, v in levels if s <= cap)
        med.append(ns * (1 + r.gauss(0, rel_noise)))
        se.append(ns * rel_noise)
    return med, se


class Stats(unittest.TestCase):
    def test_summarise(self):
        s = probe.summarise([1.0, 2.0, 3.0])
        self.assertEqual(s["median"], 2.0)
        self.assertAlmostEqual(s["se"], 1 / math.sqrt(3))

    def test_single_rep_has_no_fake_noise(self):
        s = probe.summarise([5.0])
        self.assertIsNone(s["se"]); self.assertIsNone(s["cv"])


class Levels(unittest.TestCase):
    sizes = probe.chase_sizes(256 << 20)

    def levels(self, med, se, sizes=None):
        return [l for l in probe.describe_levels(sizes or self.sizes, med, se) if l["kind"] == "level"]

    # Level detection is statistical, so these tests are too: over 200 noisy curves (fixed
    # seeds, deterministic) it must recover real levels most of the time and rarely invent
    # one. Missing a level is the conservative error; a phantom level is the one that overstates.
    SEEDS = range(200)

    def rates(self, truth, noise, sizes=None):
        sizes = sizes or self.sizes
        phantom = exact = false_boundary = 0
        for seed in self.SEEDS:
            med, se = synth_curve(truth, sizes, noise, seed)
            lv = self.levels(med, se, sizes)
            phantom += len(lv) > len(truth)
            exact += len(lv) == len(truth)
            false_boundary += sum("capacity_bytes_between" in l for l in lv) > len(truth) - 1
            if len(lv) == len(truth):  # when the count is right, values and brackets must be right
                for l, (cap, ns) in zip(lv, truth):
                    self.assertAlmostEqual(l["latency_ns"], ns, delta=ns * 0.05)
                    self.assertEqual(l["label"], "derived")
                    if "capacity_bytes_between" in l and cap != float("inf"):
                        lo, hi = l["capacity_bytes_between"]
                        self.assertTrue(lo <= cap < hi, (seed, lo, cap, hi))
        n = len(self.SEEDS)
        return phantom / n, exact / n, false_boundary / n

    def test_recovers_three_levels(self):
        ph, ex, fb = self.rates([(48 << 10, 1.8), (2 << 20, 6.0), (float("inf"), 110.0)], 0.02)
        self.assertLessEqual(ph, 0.05); self.assertLessEqual(fb, 0.05); self.assertGreaterEqual(ex, 0.9)

    def test_no_sizes_assumed(self):
        # an odd hierarchy (L1 at 100 KiB, L2 at 30 MiB) is found where it is
        ph, ex, fb = self.rates([(100 << 10, 1.0), (30 << 20, 20.0), (float("inf"), 90.0)], 0.02)
        self.assertLessEqual(ph, 0.05); self.assertLessEqual(fb, 0.05); self.assertGreaterEqual(ex, 0.9)

    def test_flat_curve_is_one_level(self):
        # pure noise around one latency: false splits at most 5% of the time
        # pure noise: measured 4.5% phantom level, 7.5% false boundary on a 12-point curve
        ph, ex, fb = self.rates([(float("inf"), 1.8)], 0.05, self.sizes[:12])
        self.assertLessEqual(ph, 0.05); self.assertLessEqual(fb, 0.08); self.assertGreaterEqual(ex, 0.9)

    def test_slope_is_not_a_staircase(self):
        # verifier's case: smooth 3%/step growth must not become a stack of levels
        med = [10 * 1.03 ** i for i in range(len(self.sizes))]
        se = [m * 0.005 for m in med]
        self.assertEqual(self.levels(med, se), [])

    def test_short_curve_is_not_averaged(self):
        # verifier's case: [1, 100] used to come out as one level at 50.5 ns
        out = probe.describe_levels([4096, 8192], [1.0, 100.0], [0.01, 1.0])
        self.assertFalse(any(abs(l.get("latency_ns", 0) - 50.5) < 5 for l in out))
        self.assertEqual(len(out), 1)  # nothing spans 2x and is flat: one transition
        self.assertEqual(out[0]["kind"], "transition")

    def test_no_noise_means_no_levels(self):
        # one rep: no SE, so no fit (a fit would invent or hide levels)
        self.assertIsNone(probe.describe_levels(self.sizes[:4], [1, 1, 2, 2], [None] * 4))


class Tiers(unittest.TestCase):
    def test_rapl_absent(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(probe.tier_rapl(root)["available"])

    def test_rapl_readable(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "sys/class/powercap/intel-rapl:0"); os.makedirs(d)
            put(os.path.join(d, "energy_uj"), "123456\n")
            t = probe.tier_rapl(root)
            self.assertTrue(t["available"]); self.assertEqual(len(t["zones"]), 1)

    def test_rapl_present_but_not_numeric(self):
        # what an unreadable zone looks like when read fails or returns junk
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "sys/class/powercap/intel-rapl:0"); os.makedirs(d)
            os.mkdir(os.path.join(d, "energy_uj"))  # a directory: read() fails even as root
            t = probe.tier_rapl(root)
            self.assertFalse(t["available"]); self.assertIn("not readable", t["evidence"])

    def test_every_tier_has_evidence(self):
        for t in probe.tiers():
            self.assertIn("evidence", t); self.assertIn(t["tier"], (1, 2, 3))
        self.assertFalse(next(t for t in probe.tiers() if t["tier"] == 3)["available"])


class Reported(unittest.TestCase):
    def test_parse_sysfs_caches(self):
        with tempfile.TemporaryDirectory() as root:
            for i, (lvl, typ, size) in enumerate([(1, "Data", "48K"), (2, "Unified", "2048K"), (3, "Unified", "260M")]):
                d = os.path.join(root, f"sys/devices/system/cpu/cpu0/cache/index{i}"); os.makedirs(d)
                for k, v in {"level": lvl, "type": typ, "size": size, "coherency_line_size": 64, "shared_cpu_list": "0"}.items():
                    put(os.path.join(d, k), f"{v}\n")
            c = probe.reported_caches(root)
            self.assertEqual([x["size_bytes"] for x in c], [48 << 10, 2 << 20, 260 << 20])
            self.assertTrue(all(x["label"] == "reported" for x in c))


class EndToEnd(unittest.TestCase):
    """A real (tiny) run: profile is strict JSON, labelled, and recomputable from raw;
    tampering with one raw number is caught by --reanalyse."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.td.name, "p.json")
        subprocess.run([sys.executable, os.path.join(HERE, "probe.py"), "--quick", "--reps", "2", "--min-ms", "2",
                        "--max-bytes", str(1 << 20), "--seed", "1", "--out", cls.path],
                       check=True, capture_output=True)
        with open(cls.path) as f:  # strict JSON: NaN/Infinity would raise here
            cls.p = json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_labels(self):
        for v in list(self.p["bandwidth"].values()) + list(self.p["compute"].values()):
            self.assertEqual(v["label"], "measured")
        for v in self.p["derived"].values():
            self.assertEqual(v["label"], "derived")
        for c in self.p["reported_caches"]:
            self.assertEqual(c["label"], "reported")

    def test_reanalyse_matches(self):
        r = subprocess.run([sys.executable, os.path.join(HERE, "probe.py"), "--reanalyse", self.path], capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def tamper(self, f):
        with open(self.path) as fh:
            p = json.load(fh)
        f(p)
        bad = os.path.join(self.td.name, "bad.json")
        put(bad, json.dumps(p))
        return subprocess.run([sys.executable, os.path.join(HERE, "probe.py"), "--reanalyse", bad],
                              capture_output=True).returncode

    def test_settings_reps_mismatch_caught(self):
        self.assertEqual(self.tamper(lambda p: p["settings"].update(reps=70)), 1)

    def test_rate_inconsistent_with_work_caught(self):
        def f(p):  # change a rate AND every statistic derived from it: still caught by work/time
            p["raw"]["flops:vec_f64:1"][0]["GFLOPs"] *= 1.5
            new = probe.analyse(p["raw"], p["environment"], p["reported_caches"], p["meter_tiers"],
                                p["build"], p["settings"], p["elapsed_s"])
            new["created_utc"] = p["created_utc"]; p.clear(); p.update(new)
        self.assertEqual(self.tamper(f), 1)

    def test_tamper_detected(self):
        with open(self.path) as f:
            p = json.load(f)
        p["raw"]["flops:vec_f64:1"][0]["GFLOPs"] *= 1.5
        bad = os.path.join(self.td.name, "bad.json")
        put(bad, json.dumps(p))
        r = subprocess.run([sys.executable, os.path.join(HERE, "probe.py"), "--reanalyse", bad], capture_output=True)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
