"""Tests for forge-probe. Stdlib unittest:  python3 -m unittest -v test_probe"""
import json, math, os, random, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def put(path, text):
    with open(path, "w") as f:
        f.write(text)


def synth_reps(ns, reps, r, noise):
    """One point's reps. noise: 'lognormal' (3%) or 'heavy' (2% + 10% outliers at 1.5-3x),
    after the second independent review's generator."""
    out = []
    for _ in range(reps):
        if noise == "lognormal":
            v = ns * math.exp(r.gauss(0, 0.03))
        else:
            v = ns * math.exp(r.gauss(0, 0.02))
            if r.random() < 0.10:
                v *= r.uniform(1.5, 3.0)
        out.append(v)
    return out


def synth_curve(truth, sizes, reps, noise, seed):
    """Goes through the tool's own path: reps -> summarise -> (median, bootstrap SE, max)."""
    r = random.Random(seed)
    med, se, mx, mn = [], [], [], []
    for s in sizes:
        ns = next(v for cap, v in truth if s <= cap)
        st = probe.summarise(synth_reps(ns, reps, r, noise))
        med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
    return med, se, mx, mn


def level_rates(truth, sizes, reps, noise, seeds=range(100)):
    """phantom: more levels than truth; exact: same count with right values and brackets;
    false_boundary: more capacity claims than real boundaries; wrong: right count, wrong bracket."""
    ph = ex = fb = wrong = 0
    for seed in seeds:
        med, se, mx, mn = synth_curve(truth, sizes, reps, noise, seed)
        lv = [l for l in probe.describe_levels(sizes, med, se, df=reps - 1, rep_max=mx, rep_min=mn)
              if l["kind"] == "level"]
        ph += len(lv) > len(truth)
        fb += sum("capacity_bytes_between" in l for l in lv) > len(truth) - 1
        if len(lv) == len(truth):
            ok = True
            for l, (cap, ns) in zip(lv, truth):
                ok &= abs(l["latency_ns"] - ns) <= 0.1 * ns
                if "capacity_bytes_between" in l and cap != float("inf"):
                    lo, hi = l["capacity_bytes_between"]
                    ok &= lo <= cap < hi
            ex += ok; wrong += not ok
    n = len(seeds)
    return {"phantom": ph / n, "exact": ex / n, "false_boundary": fb / n, "wrong": wrong / n}


class Stats(unittest.TestCase):
    def test_summarise(self):
        s = probe.summarise([1.0, 2.0, 3.0])
        self.assertEqual(s["median"], 2.0)
        self.assertAlmostEqual(s["se"], 1 / math.sqrt(3))

    def test_single_rep_has_no_fake_noise(self):
        s = probe.summarise([5.0])
        self.assertIsNone(s["se"]); self.assertIsNone(s["cv"])


class Levels(unittest.TestCase):
    """Level detection is statistical, so these tests are: rates over 100 fixed seeds,
    through the tool's real path (reps -> bootstrap SE -> fit). Missing a level is the
    conservative error; phantom levels and false capacity boundaries overstate."""
    sizes = probe.chase_sizes(256 << 20)
    FOUR = [(24 << 10, 1.2), (768 << 10, 4.5), (24 << 20, 25.0), (float("inf"), 95.0)]

    def test_four_levels_lognormal_7reps(self):
        r = level_rates(self.FOUR, self.sizes, 7, "lognormal")
        self.assertLessEqual(r["phantom"], 0.05, r); self.assertLessEqual(r["false_boundary"], 0.05, r)
        self.assertEqual(r["wrong"], 0, r); self.assertGreaterEqual(r["exact"], 0.7, r)

    def test_four_levels_heavy_tailed_7reps(self):
        r = level_rates(self.FOUR, self.sizes, 7, "heavy")
        self.assertLessEqual(r["phantom"], 0.05, r); self.assertLessEqual(r["false_boundary"], 0.05, r)
        self.assertEqual(r["wrong"], 0, r)

    def test_quick_3reps_does_not_invent(self):
        r = level_rates(self.FOUR, self.sizes[::2], 3, "lognormal")
        self.assertLessEqual(r["phantom"], 0.05, r); self.assertLessEqual(r["false_boundary"], 0.05, r)

    def test_flat_curve(self):
        for pts in (12, len(self.sizes)):
            r = level_rates([(float("inf"), 1.8)], self.sizes[:pts], 7, "lognormal")
            self.assertLessEqual(r["phantom"], 0.05, (pts, r)); self.assertLessEqual(r["false_boundary"], 0.05, (pts, r))

    def test_gentle_ramp_claims_no_levels(self):
        # review's case: 1.5%/step growth under 5% noise came out as 4 levels; later as a lone
        # "level" with a capacity claim in 11.5% of runs. Measured now (200 seeds): 0% any level.
        any_level = boundary = 0
        for seed in range(100):
            r = random.Random(seed)
            med, se, mx, mn = [], [], [], []
            for i in range(len(self.sizes)):
                st = probe.summarise([10 * 1.015 ** i * math.exp(r.gauss(0, 0.05)) for _ in range(7)])
                med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
            lv = [l for l in probe.describe_levels(self.sizes, med, se, df=6, rep_max=mx, rep_min=mn)
                  if l["kind"] == "level"]
            any_level += bool(lv); boundary += any("capacity_bytes_between" in l for l in lv)
        self.assertLessEqual(any_level / 100, 0.08); self.assertLessEqual(boundary / 100, 0.06)

    def test_drifting_precise_levels_are_found(self):
        # regression: real plateaus drift (L2 here rose 5.6 -> 6.2 ns across its span) and 7 reps
        # measure them precisely; an absolute "zero slope" test rejected every level on real data
        hits = 0
        for seed in range(100):
            r = random.Random(seed)
            med, se, mx, mn = [], [], [], []
            for s in self.sizes:
                lx = math.log(s)
                ns = (1.8 * (1 + 0.01 * (lx - 8)) if s <= 44 << 10 else
                      5.6 * (1 + 0.04 * (lx - 11)) if s <= 1448 << 10 else
                      32 * (1 + 0.03 * (lx - 15)) if s <= 11 << 20 else 115 * (1 + 0.005 * (lx - 17)))
                st = probe.summarise([ns * math.exp(r.gauss(0, 0.01)) for _ in range(7)])
                med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
            lv = [l for l in probe.describe_levels(self.sizes, med, se, df=6, rep_max=mx, rep_min=mn)
                  if l["kind"] == "level"]
            hits += len(lv) == 4
        self.assertGreaterEqual(hits / 100, 0.9)

    def test_shared_vm_curve_keeps_clean_levels(self):
        # regression from independent verification: on a noisy shared host the old fit lost L1 and L2
        # (a pairwise check cascaded) and stretched a capacity bracket to the end of the sweep
        with open(os.path.join(HERE, "fixtures", "shared_vm_curve.json")) as f:
            fx = json.load(f)
        out = probe.describe_levels(fx["bytes"], fx["median"], fx["se_median"], df=fx["df"],
                                    rep_max=fx["max"], rep_min=fx["min"])
        lv = [l for l in out if l["kind"] == "level"]
        self.assertEqual(len(lv), 2, out)
        self.assertAlmostEqual(lv[0]["latency_ns"], 1.40, delta=0.05)
        self.assertAlmostEqual(lv[1]["latency_ns"], 4.48, delta=0.1)
        self.assertEqual(lv[0]["capacity_bytes_between"], [44 << 10, 64 << 10])
        self.assertIsNone(lv[1]["capacity_bytes_between"][1])  # next level never found: open, not "end of sweep"

    def test_interference_spike_is_excluded_not_a_level(self):
        sizes = self.sizes
        r = random.Random(5)
        med, se, mx, mn = [], [], [], []
        k = next(i for i, s in enumerate(sizes) if s >= 4 << 20)
        for i, s in enumerate(sizes):
            ns = 1.8 if s <= 32 << 10 else 6.0 if s <= 1 << 20 else 34.0 if s <= 16 << 20 else 117.0
            st = probe.summarise([ns * (3.0 if i == k else 1.0) * math.exp(r.gauss(0, 0.02)) for _ in range(7)])
            med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
        self.assertEqual(probe.nonmonotone_points(med, se, 6), [k])
        lv = [l for l in probe.describe_levels(sizes, med, se, df=6, rep_max=mx, rep_min=mn) if l["kind"] == "level"]
        self.assertEqual(len(lv), 4)

    def test_too_disturbed_curve_is_refused(self):
        sizes = self.sizes[:20]
        med = [10.0 if i % 2 == 0 else 30.0 for i in range(20)]  # half the points slower than the next size
        self.assertIsNone(probe.describe_levels(sizes, med, [m * 0.01 for m in med], df=6))

    def test_bump_does_not_make_extra_levels(self):
        # independent verifier's case D: a raised block inside a ramp (non-monotone)
        extra = 0
        for seed in range(100):
            r = random.Random(seed)
            med, se, mx, mn = [], [], [], []
            for i in range(len(self.sizes)):
                ns = 25.0 if 10 <= i <= 13 else 5.0 * 1.025 ** i
                st = probe.summarise([ns * math.exp(r.gauss(0, 0.03)) for _ in range(7)])
                med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
            out = probe.describe_levels(self.sizes, med, se, df=6, rep_max=mx, rep_min=mn) or []
            extra += len([l for l in out if l["kind"] == "level"]) > 1
        self.assertLessEqual(extra / 100, 0.05)

    def test_curved_precise_ramp_has_no_levels(self):
        # fifth review: latency = 2 + 3.5 i ns (curved in log-log) at 0.2% noise gave a phantom level
        # with a capacity claim in 51.5% of draws, because the edge test ignored the jump's direction
        bad = 0
        for seed in range(100):
            r = random.Random(seed)
            med, se, mx, mn = [], [], [], []
            for i in range(len(self.sizes)):
                st = probe.summarise([(2 + 3.5 * i) * math.exp(r.gauss(0, 0.002)) for _ in range(7)])
                med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
            out = probe.describe_levels(self.sizes, med, se, df=6, rep_max=mx, rep_min=mn) or []
            bad += any(l["kind"] == "level" for l in out)
        self.assertLessEqual(bad / 100, 0.05)

    def test_extreme_weights_do_not_crash(self):
        sizes = self.sizes[:20]
        med = [1.0 + 0.01 * i for i in range(20)]
        se = [1e-12 if i % 2 else 1e-2 for i in range(20)]
        probe.describe_levels(sizes, med, se, df=6)  # must not raise
        self.assertIsNone(probe.describe_levels(sizes, [0.0] + med[1:], se, df=6))

    def test_slope_is_not_a_staircase(self):
        med = [10 * 1.03 ** i for i in range(len(self.sizes))]
        self.assertEqual([l for l in probe.describe_levels(self.sizes, med, [m * 0.005 for m in med], df=6)
                          if l["kind"] == "level"], [])

    def test_mixed_boundary_point_moves_bracket(self):
        # second review's case: a point where 3 of 7 reps already run at DRAM speed sits inside
        # the L3 level by median; the capacity bracket must start before it
        sizes = self.sizes
        r = random.Random(3)
        med, se, mx, mn = [], [], [], []
        k_mixed = next(i for i, s in enumerate(sizes) if s >= 12 << 20)
        for i, s in enumerate(sizes):
            ns = 1.8 if s <= 32 << 10 else 34.0 if s < 16 << 20 else 117.0
            xs = [ns * math.exp(r.gauss(0, 0.02)) for _ in range(7)]
            if i == k_mixed:
                xs = [34.0] * 4 + [95.0] * 3
            st = probe.summarise(xs)
            med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
        l3 = [l for l in probe.describe_levels(sizes, med, se, df=6, rep_max=mx, rep_min=mn)
              if l["kind"] == "level"][1]
        self.assertLess(l3["capacity_bytes_between"][0], sizes[k_mixed])

    def test_mixed_point_absorbed_upward_moves_bracket_up(self):
        # review's case: a point still at L3 speed in some reps, absorbed into DRAM by its median;
        # the bracket's upper end must not land before it
        sizes = self.sizes
        r = random.Random(4)
        med, se, mx, mn = [], [], [], []
        k = max(i for i, s in enumerate(sizes) if s < 24 << 20)  # last size truly within L3
        for i, s in enumerate(sizes):
            ns = 1.8 if s <= 32 << 10 else 34.0 if s < 24 << 20 else 117.0
            xs = [ns * math.exp(r.gauss(0, 0.02)) for _ in range(7)]
            if i == k:
                xs = [117.0] * 4 + [34.0] * 3
            st = probe.summarise(xs)
            med.append(st["median"]); se.append(st["se_median"]); mx.append(st["max"]); mn.append(st["min"])
        l3 = [l for l in probe.describe_levels(sizes, med, se, df=6, rep_max=mx, rep_min=mn)
              if l["kind"] == "level"][1]
        self.assertGreater(l3["capacity_bytes_between"][1], sizes[k])

    def test_short_curve_is_not_averaged(self):
        out = probe.describe_levels([4096, 8192], [1.0, 100.0], [0.01, 1.0], df=6)
        self.assertEqual([l["kind"] for l in out], ["transition"])

    def test_no_noise_means_no_levels(self):
        self.assertIsNone(probe.describe_levels(self.sizes[:4], [1, 1, 2, 2], [None] * 4))

    def test_t_quantile(self):
        self.assertAlmostEqual(probe.t_quantile(0.975, 6), 2.447, places=2)
        self.assertAlmostEqual(probe.t_quantile(0.975, 2), 4.303, places=2)
        self.assertAlmostEqual(probe.t_quantile(0.975, None), 1.960, places=2)


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
            if v is not None:  # null when the data cannot support the figure
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

    def test_off_host_is_not_a_clean_pass(self):
        # review's bypass: change the host id and the meter claim went unchecked with exit 0
        def f(p):
            p["environment"]["hostname_hash"] = "000000000000"
            for m in p["meter_tiers"]:
                if m["name"] == "rapl_energy":
                    m["available"] = True
            p["best_tier_for_energy"] = "rapl_energy"
            probe.add_digests(p)  # a careful forger recomputes the digest too
        self.assertEqual(self.tamper(f), 2)

    def test_edit_without_new_digest_caught(self):
        def f(p):
            p["environment"]["hostname_hash"] = "000000000000"
        self.assertEqual(self.tamper(f), 1)

    def test_calibration_claim_flip_caught(self):
        def f(p):
            next(m for m in p["meter_tiers"] if m["tier"] == 3)["available"] = True
        self.assertEqual(self.tamper(f), 1)

    def test_short_timed_window_caught(self):
        def f(p):
            rec = p["raw"]["flops:vec_f64:1"][0]
            rec["seconds"] /= 10
            rec["GFLOPs"] *= 10  # keep rate = work/time consistent; only the window is too short
        self.assertEqual(self.tamper(f), 1)

    def test_build_records_vector_isa(self):
        self.assertIn(self.p["build"]["vector_isa"], ("avx512", "avx2", "avx", "sve", "neon", "sse2", "scalar-emulated"))

    def test_digest_present_and_reproducible(self):
        d = self.p["digests"]
        self.assertEqual(len(d["profile_sha256"]), 64)
        body = {k: v for k, v in self.p.items() if k != "digests"}
        import hashlib
        self.assertEqual(hashlib.sha256(probe.canonical(body)).hexdigest(), d["profile_sha256"])

    def test_digest_changes_with_any_edit(self):
        p = json.loads(json.dumps(self.p))
        p["settings"]["note"] = "x"
        self.assertNotEqual(probe.add_digests(p)["digests"]["profile_sha256"], self.p["digests"]["profile_sha256"])

    def test_canonical_has_no_join_collision(self):
        # the '|'-join bug class: moving a separator between fields must change the bytes
        self.assertNotEqual(probe.canonical({"a": "x|y", "b": "z"}), probe.canonical({"a": "x", "b": "y|z"}))

    def test_meter_claim_flip_caught_on_same_host(self):
        def f(p):
            for m in p["meter_tiers"]:
                if m["name"] == "rapl_energy":
                    m["available"] = not m["available"]
            p["best_tier_for_energy"] = "rapl_energy"
        self.assertEqual(self.tamper(f), 1)

    def test_zero_seconds_caught(self):
        def f(p):
            p["raw"]["chase:4096"][0]["seconds"] = 0
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
