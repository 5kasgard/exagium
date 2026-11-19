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
  - accuracy on each segment of pool images past the first 956 (956-4,096, 4,096-16,384, ...) must not fall
    below accuracy on the first 956 by more than 3 SE: a candidate that does less work at large N (reviewer's test, 2026-11-18: classify 1,000 images,
    print 7 for the rest) would otherwise look cheap. Both ranges come from the same shuffle, so they are
    comparable to each other; pool images come from training rows, so they are not an accuracy estimate.

What the per-image figure includes: everything the candidate does per image, including how it stages its
input (one big buffer, blocks, mmap). That is part of the candidate. Every candidate gets the same file, x.bin.

Not measured: block steps. One extra image costs 0.03-0.3 us; single runs vary by 75-3,800 us here.

Reuses loop.py (frozen by plan 2) by import; changes nothing in it. Candidate C is compiled inside the jail by
runner/jail.py (queue Q56b), never on the host; only the trusted timer is built on the host.
Usage: python3 loop/scale.py DIR... [--sizes 0,956,4096,16384,65536] [--rounds 9] [--seed 1] [--out F]
       [--provenance TEXT]   (where candidates outside this repo came from: branch and commit)
Exit codes: 0 done, no problems; 1 a check failed or problems were found; 3 failure.
"""
import argparse, json, math, os, random, shutil, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I drops the script's directory
import loop as L  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runner"))
import jail as J  # noqa: E402  (Q56b: compile candidates in the jail; imports loop itself, the same module)

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


def accuracy_check(pred, labels, sizes, first=TASK_N):
    """Accuracy on the first `first` images, and on each later segment between run sizes; a problem if any later
    segment is worse than the first by more than 3 SE (a candidate doing less work somewhere past N = 956)."""
    def acc(p, y):
        a = sum(x == z for x, z in zip(p, y)) / len(y)
        return a, math.sqrt(a * (1 - a) / len(y))
    base = acc(pred[:first], labels[:first])
    rec = {"first": first, "accuracy_first": base[0], "segments": [], "problem": None}
    edges = [first] + [n for n in sizes if n > first]
    for a, b in zip(edges, edges[1:]):
        seg = acc(pred[a:b], labels[a:b])
        rec["segments"].append({"from": a, "to": b, "accuracy": seg[0]})
        if seg[0] < base[0] - 3 * math.hypot(base[1], seg[1]) and rec["problem"] is None:
            rec["problem"] = (f"accuracy on images {a}-{b} is {seg[0]:.3f} against {base[0]:.3f} on the first {first}: "
                              "it may do less work at larger N")
    rec["accuracy_rest"] = acc(pred[first:], labels[first:])[0] if len(labels) > first else None
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
               "marginal_ns_per_image_large_n_median": statistics.median([marginal(r) for r in rounds]),
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
            J.compile_candidate(src, c["exe"])   # Q56b: never compile candidate code on the host
            c["ref"], _ = L.timed(c, imgs, timer, cpu)  # reference predictions on the whole pool (not a timing)
            acc[name] = accuracy_check(c["ref"], labels, sizes)
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
        J.check()   # loop's jail mode, then fail closed unless limits hold and the jailed compiler refuses host files
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
