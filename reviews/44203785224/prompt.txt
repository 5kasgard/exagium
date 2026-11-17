You are reviewing a measurement tool and the claims made from its first results. Be adversarial and specific. Do not praise. For each problem give: location (function or README line), what is wrong, why it matters, and a concrete fix. Grade each critical / major / minor. End with a verdict: merge / merge after fixes / do not merge.

CONTEXT
- A harness (loop.py, frozen, not shown) scores small C programs ("candidates") that classify 8x8 digit images: ./infer WEIGHTS X N reads N images of 64 bytes (pixels 0..16) from file X and prints N digits. A trusted timer runs each candidate in a jail and reports wall-clock ns for the whole process.
- The old meter is (t(956) - t(0)) / 956, where about 1 ms of process start-up dominates a ~0.1 ms workload.
- The new tool, scale.py, times every candidate at sizes 0, 956, 4096, 16384, 65536 images, in rounds (each round: every candidate at every size, shuffled order), and reports t(0), the slope between the two largest sizes ("large-N marginal cost per image"), every segment slope, and pairwise differences paired by round. Inputs: distinct images from training rows only (originals plus copies with 4 random pixels moved by +-1). Checks: predictions must match a reference run on the same images; accuracy beyond the first 956 images must not fall more than 3 SE below the first 956 (catches a candidate that does less work at large N).
- An earlier version was reviewed; it was fixed for: a cheating candidate passing; a misnamed OLS fit; shifted inputs that destroyed accuracy (1-pixel shift: 0.99 -> ~0.5); unmeasurable block-step claims.

QUESTIONS TO ATTACK
1. Is the large-N slope (two sizes) a sound estimator? Bias, variance, outliers (one round had t(0)=8.5 ms vs ~1 ms; does it affect the large-N slope?). Should it use medians or a robust fit?
2. Is pairing by round valid given the shuffled order within rounds? Is t(14) at 2.145 the right threshold for 15 rounds?
3. Does the nudged-image pool make cost representative? Could near-duplicate images (4 pixels changed) favour candidates with caches or branch prediction?
4. Does the accuracy check have enough power? Could a candidate cheat in a way it misses (e.g. cheat only past 16,384 images, or degrade gracefully)?
5. Are the README claims supported by the results shown? Anything overstated?

FILES FOLLOW.

===== loop/scale.py =====
"""Per-image cost at large run sizes, next to the fixed cost per run (2026-11-18).

Why: loop.py's meter is (t(N) - t(0)) / N at the task's N = 956. At that size about 1 ms of process start-up
dominates, and pilot 2 found the ranking changes at 95,600 images (HANDOFF-lgn-pilot2, next item 1). This tool
times each candidate, in loop.py's jail with loop.py's trusted timer, at several run sizes, and reports:
  - fixed cost per run: the measured t(0);
  - marginal cost per image at large N: (t(N_max) - t(N_prev)) / (N_max - N_prev), the two largest sizes;
  - the slope of every segment between consecutive sizes, because cost per image is not constant in N
    (first run: lgn_d's slope was 98 ns over 0-4,096 images and 53 ns over 16,384-65,536).
Rounds are the unit of replication: within a round every candidate runs at every size, in a shuffled order,
so differences between candidates are paired by round. Intervals with few rounds should use t, not 2.

Inputs: DISTINCT images made from the training rows only (the first 3/4 of optdigits.tra; never the dev rows,
never the test file): each original row, plus copies with 4 random pixels moved by +-1 (clipped to 0..16); each
keeps its source label. Run size N uses the first N images of one seeded shuffle of the pool. Not shifted
copies: a 1-pixel shift drops mlp16 from 0.99 to 0.44-0.56 accuracy (2 pixels: 0.12-0.17), and an MLP's cost
depends on which hidden units fire, so off-distribution inputs could misstate cost; nudged copies keep 0.99.

Checks (each failure is listed under "problems", exit 1):
  - every timed run's predictions equal the same images' predictions in a reference run on the whole pool;
  - accuracy on the pool images beyond the first 956 must not fall below accuracy on the first 956 by more
    than 3 SE: a candidate that does less work at large N (reviewer's test, 2026-11-18: classify 1,000 images,
    print 7 for the rest) would otherwise look cheap. Both ranges come from the same shuffle, so they are
    comparable to each other; pool images come from training rows, so they are not an accuracy estimate.

What the per-image figure includes: everything the candidate does per image, including how it stages its
input (one big buffer, blocks, mmap). That is part of the candidate. Every candidate gets the same file, x.bin.

Not measured: block steps. One extra image costs 0.03-0.3 us; single runs vary by 75-3,800 us here.

Reuses loop.py (frozen by plan 2) by import; changes nothing in it.
Usage: python3 loop/scale.py DIR... [--sizes 0,956,4096,16384,65536] [--rounds 9] [--seed 1] [--out F]
       [--provenance TEXT]   (where candidates outside this repo came from: branch and commit)
Exit codes: 0 done, no problems; 1 a check failed or problems were found; 3 failure.
"""
import argparse, json, math, os, random, shutil, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I drops the script's directory
import loop as L  # noqa: E402

DEFAULT_SIZES = (0, 956, 4096, 16384, 65536)
NUDGES = 4
TASK_N = 956


def nudged(row, rng, k=NUDGES):
    out = list(row)
    for _ in range(k):
        i = rng.randrange(64)
        out[i] = min(16, max(0, out[i] + rng.choice((-1, 1))))
    return tuple(out)


def pool(n, seed):
    """(images, labels): n distinct 64-pixel images from the training rows (never dev or test), one seeded shuffle."""
    tr = L.load(L.TASK["train"])
    train = tr[:len(tr) * 3 // 4]  # loop.py's dev split is tr[len*3//4:]; these are the rows before it
    rng = random.Random(seed)
    seen, imgs = set(), []
    for r in train:
        im = tuple(r[:64])
        if im not in seen:
            seen.add(im)
            imgs.append((im, r[64]))
    tries = 0
    while len(imgs) < n:
        tries += 1
        if tries > 50 * n:
            raise L.Failed(f"only {len(imgs)} distinct images could be made; asked for {n}")
        r = train[tries % len(train)]
        im = nudged(r[:64], rng)
        if im not in seen:
            seen.add(im)
            imgs.append((im, r[64]))
    random.Random(seed).shuffle(imgs)
    return [list(im) for im, _ in imgs[:n]], [y for _, y in imgs[:n]]


def accuracy_check(pred, labels, split=TASK_N):
    """Accuracy on the first `split` images and on the rest; a problem if the rest is worse by more than 3 SE."""
    def acc(p, y):
        a = sum(x == z for x, z in zip(p, y)) / len(y)
        return a, math.sqrt(a * (1 - a) / len(y))
    small = acc(pred[:split], labels[:split])
    rec = {"first": split, "accuracy_first": small[0], "accuracy_rest": None, "problem": None}
    if len(labels) > split:
        rest = acc(pred[split:], labels[split:])
        rec["accuracy_rest"] = rest[0]
        if rest[0] < small[0] - 3 * math.hypot(small[1], rest[1]):
            rec["problem"] = (f"accuracy beyond the first {split} images is {rest[0]:.3f} against {small[0]:.3f} "
                              "on the first: it may do less work at large N")
    return rec


def summarise(ts, sizes):
    """ts: {name: [ {N: ns} per round ]}, sizes sorted. Per-candidate and pairwise summaries."""
    big, prev = sizes[-1], sizes[-2]

    def marginal(r):
        return (r[big] - r[prev]) / (big - prev)
    out = {}
    for name, rounds in ts.items():
        rec = {"fixed_ns_t0": L.mean_se([r[sizes[0]] for r in rounds]) if sizes[0] == 0 else None,
               "marginal_ns_per_image_large_n": L.mean_se([marginal(r) for r in rounds]),
               "large_n_segment": [prev, big],
               "segments": [{"from": a, "to": b, "ns_per_image": L.mean_se([(r[b] - r[a]) / (b - a) for r in rounds])}
                            for a, b in zip(sizes, sizes[1:])],
               "per_size": {str(n): {"run_ns": L.mean_se([r[n] for r in rounds]),
                                     "ns_per_image": L.mean_se([r[n] / n for r in rounds]) if n else None}
                            for n in sizes}}
        if TASK_N in sizes and 0 in sizes:  # loop.py's formula, on pool images (not dev rows)
            rec["n956_pool_minus_t0_ns_per_image"] = L.mean_se([(r[TASK_N] - r[0]) / TASK_N for r in rounds])
        out[name] = rec
    names = sorted(ts)
    pairs = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            mx = [marginal(r) for r in ts[x]]
            my = [marginal(r) for r in ts[y]]
            m, se = L.mean_se([p - q for p, q in zip(mx, my)])  # paired by round index
            p = {"a": x, "b": y, "marginal_diff_ns": m, "marginal_diff_se": se,
                 "ratio_of_means": statistics.fmean(mx) / statistics.fmean(my)}
            if TASK_N in sizes and 0 in sizes:
                m2, se2 = L.mean_se([(r[TASK_N] - r[0]) / TASK_N - (s[TASK_N] - s[0]) / TASK_N
                                     for r, s in zip(ts[x], ts[y])])
                p["n956_diff_ns"], p["n956_diff_se"] = m2, se2
            pairs.append(p)
    return out, pairs


def measure(dirs, sizes, rounds, seed):
    if L.JAIL["mode"] not in ("bwrap", "confine"):
        raise L.Failed("no jail selected: call loop.check_jail() first")
    names = [os.path.basename(d.rstrip("/")) for d in dirs]
    if len(set(names)) != len(names):
        raise L.Failed(f"candidate names must be unique: {names}")
    imgs, labels = pool(max(sizes), seed)
    rng = random.Random(seed)
    cpu = min(os.sched_getaffinity(0))
    work = tempfile.mkdtemp(prefix="exagium-scale-")
    problems, acc = [], {}
    try:
        timer = os.path.join(work, "timer")
        L.build(L.TIMER, timer)
        cands = []
        for d, name in zip(dirs, names):
            os.makedirs(os.path.join(work, name))
            src = os.path.join(work, name, "infer.c")
            shutil.copyfile(os.path.join(d, "infer.c"), src)
            c = {"name": name, "dir": d, "exe": os.path.join(work, name, "infer")}
            L.build(src, c["exe"])
            c["ref"], _ = L.timed(c, imgs, timer, cpu)  # reference predictions on the whole pool (not a timing)
            acc[name] = accuracy_check(c["ref"], labels)
            if acc[name]["problem"]:
                problems.append(f"{name}: {acc[name]['problem']}")
            cands.append(c)
        ts = {c["name"]: [] for c in cands}
        for _ in range(rounds):
            jobs = [(c, n) for c in cands for n in sizes]
            rng.shuffle(jobs)
            t = {c["name"]: {} for c in cands}
            for c, n in jobs:
                pred, t[c["name"]][n] = L.timed(c, imgs[:n], timer, cpu)
                if pred != c["ref"][:n]:
                    problems.append(f"{c['name']}: predictions at N={n} differ from the same images in the reference run")
            for c in cands:
                ts[c["name"]].append(t[c["name"]])
    finally:
        shutil.rmtree(work)
    per, pairs = summarise(ts, list(sizes))
    for name in per:
        per[name]["pool_accuracy"] = acc[name]
    return {"tool": "loop/scale.py", "sizes": list(sizes), "rounds": rounds, "seed": seed,
            "pool": {"source": "first 3/4 of optdigits.tra, originals + copies with 4 pixels moved by 1, distinct", "n": max(sizes),
                     "sha256": L.sha256_bytes(bytes(v for im in imgs for v in im))},
            "host": {"cpu": L.cpu_model(), "jail": L.JAIL["mode"], "versions": L.versions()},
            "digests": L.digests(dirs) + [{"name": L.rel(os.path.abspath(__file__)), "sha256": L.sha256_file(os.path.abspath(__file__))}],
            "candidates": per, "pairs": pairs, "raw_ns": ts, "problems": problems}


def show(s):
    print(f"sizes {s['sizes']}, {s['rounds']} rounds, jail {s['host']['jail']}, {s['host']['cpu']}")
    for name, r in sorted(s["candidates"].items(), key=lambda kv: kv[1]["marginal_ns_per_image_large_n"][0]):
        b, bse = r["marginal_ns_per_image_large_n"]
        line = f"{name:16} large-N {b:7.1f} ±{bse or 0:5.1f} ns/img"
        if r["fixed_ns_t0"]:
            line += f"   t(0) {r['fixed_ns_t0'][0] / 1e6:6.3f} ms"
        if r.get("n956_pool_minus_t0_ns_per_image"):
            m, se = r["n956_pool_minus_t0_ns_per_image"]
            line += f"   N=956 minus t(0) {m:6.1f} ±{se or 0:5.1f}"
        a = r["pool_accuracy"]
        line += f"   acc {a['accuracy_first']:.3f}/{a['accuracy_rest']:.3f}" if a["accuracy_rest"] is not None else ""
        print(line)
        print("                 segments ns/img: " + ", ".join(
            f"{g['from']}-{g['to']}: {g['ns_per_image'][0]:.0f}" for g in r["segments"]))
    for p in s["pairs"]:
        print(f"  {p['a']} - {p['b']}: large-N {p['marginal_diff_ns']:+.1f} ±{p['marginal_diff_se'] or 0:.1f} ns/img "
              f"(ratio {p['ratio_of_means']:.2f})")
    for x in s["problems"]:
        print("PROBLEM:", x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the result JSON here (never overwritten)")
    ap.add_argument("--provenance", help="where candidates outside this repo came from (branch and commit)")
    a = ap.parse_args(argv)
    try:
        sizes = sorted({int(x) for x in a.sizes.split(",")})
        if len(sizes) < 3 or sizes[0] < 0:
            ap.error("--sizes needs at least three distinct non-negative sizes")
        if a.rounds < 2:
            ap.error("--rounds must be >= 2")
        dirs = [L.candidate(d) for d in a.dirs]
        L.check_jail()
        s = measure(dirs, sizes, a.rounds, a.seed)
        s["provenance"] = a.provenance
        show(s)
        if a.out:
            with open(a.out, "x") as fh:
                json.dump(s, fh, indent=1)
            print(f"wrote {a.out} sha256 {L.sha256_file(a.out)}")
        return 1 if s["problems"] else 0
    except L.Failed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"scale failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

===== loop/test_scale.py =====
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
        self.assertIsNone(scale.accuracy_check(pred, y)["problem"])

    def test_less_work_at_large_n_is_a_problem(self):
        y = [i % 10 for i in range(5000)]
        pred = y[:1000] + [7] * 4000  # the reviewer's cheat: real answers early, a constant later
        self.assertIsNotNone(scale.accuracy_check(pred, y)["problem"])


class Measure(unittest.TestCase):
    """measure() with the jail, compiler and timer faked: the checks must fire."""

    def run_measure(self, behaviour):
        calls = {"n": 0}

        def fake_timed(c, rows, timer, cpu):
            calls["n"] += 1
            return behaviour(rows, calls["n"]), 1_000_000 + 50 * len(rows)

        with mock.patch.object(L, "build", lambda src, out: None), \
             mock.patch.object(L, "timed", fake_timed), \
             mock.patch.dict(L.JAIL, {"mode": "confine"}), \
             mock.patch.object(L, "versions", lambda: {}):
            return scale.measure([MLP16], [0, 956, 4096], 2, 1)

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
        self.assertTrue(any("less work at large N" in p for p in s["problems"]))

    def test_refuses_without_a_jail(self):
        with mock.patch.dict(L.JAIL, {"mode": None}):
            with self.assertRaises(L.Failed):
                scale.measure([MLP16], [0, 956, 4096], 2, 1)


if __name__ == "__main__":
    unittest.main()

===== README section =====
## Cost per image at scale (`scale.py`, added 2026-11-18)
Both meters above work at the task's 956 images, where a run is about 1 ms of process start-up plus about 0.1 ms of classifying. `scale.py` times each candidate, with this file's jail and timer, at several run sizes (default 0, 956, 4,096, 16,384 and 65,536 images). It reports:
- **the fixed cost per run:** the measured t(0);
- **the marginal cost per image at large N:** the slope between the two largest sizes;
- **the slope of every segment,** because cost per image isn't constant in N.

**How the comparison stays fair:**
- Rounds are paired: every candidate runs at every size in each round, in a shuffled order.
- The images are 65,536 distinct images from the training rows only, never dev or test. Each is either an original or a copy with 4 pixels moved by ±1. Shifted copies were tried first and dropped: a 1-pixel shift cuts mlp16's accuracy from 0.99 to 0.44–0.56, and an MLP's cost depends on which hidden units fire.
- Every run's predictions must match a reference run on the same images.
- Accuracy past the first 956 images must not fall more than 3 SE below accuracy on the first 956. A copy of mlp16 that classified 1,000 images and printed 7 for the rest passed the first version unflagged; it is now caught.

**What the per-image figure includes:** everything the candidate does per image, including how it stages its input, because that is part of the candidate. The harness hands every candidate the same file.

**Run of 2026-11-18:** one host (Xeon @ 2.30 GHz), confine jail, 15 rounds, `results/scale-2026-11-18/`. `lgn_d`, `lgn_blockread` and `mlp16_chunked` come from the pilot branch; the JSON records where.

| | large-N cost per image (ns) | t(0) (ms) | N = 956 on pool images, t(0) removed (ns) |
|---|---|---|---|
| lgn_blockread | 37.9 ± 3.5 | 1.56 | -354 ± 523 |
| lgn_d | 73.0 ± 6.3 | 1.03 | 185 ± 50 |
| mlp16_chunked | 91.1 ± 8.3 | 1.08 | 136 ± 31 |
| mlp16 | 115.6 ± 2.5 | 1.13 | 116 ± 29 |
| mlp32 | 164.5 ± 10.4 | 1.13 | 152 ± 36 |
| thermo | 191.1 ± 9.1 | 1.04 | 320 ± 72 |
| linear | 209.9 ± 29.5 | 1.13 | 185 ± 57 |
| mlp64 | 250.4 ± 4.5 | 1.16 | 288 ± 72 |

- **Separation:** the large-N figure separates 23 of the 28 pairs at t(14). The N = 956 figure separates 3.
- **Cost per image changes with N.** For example, lgn_d's segments run 185, 68, 58 and 73 ns. The large-N column is a marginal cost at large N, not a constant.
- **This run against the first:** a first run (9 rounds, shifted inputs, an earlier version of the tool) put lgn_d at 53.9 ± 1.3 ns, against 73.0 ± 6.3 here. Session-to-session spread is larger than one session's SE. Read the ranking, not the decimals.
- **One noisy round:** lgn_blockread's t(0) and its N = 956 figure carry one slow round.
- **linear is slow per image for a known reason:** it runs 3,813 instructions per image (plan 2's counter) at about 20 per ns. mlp16 runs 645 at about 6 per ns.
- **Block steps** (one image past a 1,024-image block) **aren't measured.** One image costs 0.03–0.3 µs, and single runs vary by tens to thousands of µs.

This is one host and one session: direction, not a claim. A cost claim still needs the two-meter rule above on every host that scores it.


===== results summary (results/scale-2026-11-18/xeon-2.30-15rounds.txt, first 18 lines; raw ns per round in the JSON) =====
sizes [0, 956, 4096, 16384, 65536], 15 rounds, jail confine, Intel(R) Xeon(R) Processor @ 2.30GHz
lgn_blockread    large-N    37.9 ±  3.5 ns/img   t(0)  1.556 ms   N=956 minus t(0) -354.4 ±523.1   acc 0.990/0.994
                 segments ns/img: 0-956: -354, 956-4096: 13, 4096-16384: 43, 16384-65536: 38
lgn_d            large-N    73.0 ±  6.3 ns/img   t(0)  1.032 ms   N=956 minus t(0)  184.5 ± 50.0   acc 0.990/0.994
                 segments ns/img: 0-956: 185, 956-4096: 68, 4096-16384: 58, 16384-65536: 73
mlp16_chunked    large-N    91.1 ±  8.3 ns/img   t(0)  1.085 ms   N=956 minus t(0)  135.6 ± 30.5   acc 0.985/0.990
                 segments ns/img: 0-956: 136, 956-4096: 110, 4096-16384: 77, 16384-65536: 91
mlp16            large-N   115.6 ±  2.5 ns/img   t(0)  1.127 ms   N=956 minus t(0)  115.6 ± 29.1   acc 0.985/0.990
                 segments ns/img: 0-956: 116, 956-4096: 86, 4096-16384: 121, 16384-65536: 116
mlp32            large-N   164.5 ± 10.4 ns/img   t(0)  1.132 ms   N=956 minus t(0)  151.9 ± 35.8   acc 1.000/1.000
                 segments ns/img: 0-956: 152, 956-4096: 162, 4096-16384: 174, 16384-65536: 165
thermo           large-N   191.1 ±  9.1 ns/img   t(0)  1.045 ms   N=956 minus t(0)  320.2 ± 72.3   acc 1.000/1.000
                 segments ns/img: 0-956: 320, 956-4096: 174, 4096-16384: 245, 16384-65536: 191
linear           large-N   209.9 ± 29.5 ns/img   t(0)  1.127 ms   N=956 minus t(0)  185.5 ± 56.9   acc 0.984/0.986
                 segments ns/img: 0-956: 185, 956-4096: 176, 4096-16384: 232, 16384-65536: 210
mlp64            large-N   250.4 ±  4.5 ns/img   t(0)  1.161 ms   N=956 minus t(0)  288.4 ± 71.8   acc 0.996/0.996
                 segments ns/img: 0-956: 288, 956-4096: 267, 4096-16384: 239, 16384-65536: 250
  lgn_blockread - lgn_d: large-N -35.1 ±6.0 ns/img (ratio 0.52)

lgn_blockread raw t(0) ms by round: 1.15 1.02 1.0 1.46 0.95 1.03 1.14 0.94 0.99 1.0 1.08 1.03 1.04 0.99 8.52
