"""Per-image cost meter: fixed cost per run and cost per image, measured, not assumed (2026-11-18).

Why: loop.py's meter is (t(N) - t(0)) / N at the task's N = 956. At that size fixed per-run costs dominate,
and pilot 2 found the ranking changes at 95,600 images (HANDOFF-lgn-pilot2, next item 1). This tool times
each candidate, in loop.py's jail with loop.py's trusted timer, at several run sizes, and fits
    t(N) = fixed + per_image * N
per round. Rounds are the unit of replication: within a round every candidate runs at every size, in a
shuffled order, so differences between candidates are paired.

Inputs: a pool of DISTINCT images made from the training rows only (the first 3/4 of optdigits.tra; never the
dev rows, never the test file), shifted by -2..2 pixels in x and y with zero fill. Labels are not used: this
meters cost, not accuracy. Run size N uses the first N images of one seeded shuffle of the pool, so every run
of a candidate must give the same predictions on the same images; any difference is reported as a problem.

What the per-image figure includes: everything the candidate does per image, including how it stages its
input (one big buffer, blocks, mmap). That is part of the candidate. The harness supplies input the same way
to every candidate: one file, x.bin, as loop.py does.

Block steps: the residual table shows each size's mean distance from the fitted line; sizes either side of a
block boundary (1024 / 1025) show steps that a straight line hides.

Reuses loop.py (frozen by plan 2) by import; changes nothing in it.
Usage: python3 loop/scale.py DIR... [--sizes 0,956,1024,1025,4096,16384,65536] [--rounds 7] [--seed 1] [--out F]
Exit codes: 0 done, no problems; 1 a check failed or problems were found; 3 failure.
"""
import argparse, json, os, random, shutil, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I drops the script's directory
import loop as L  # noqa: E402

DEFAULT_SIZES = (0, 956, 1024, 1025, 4096, 16384, 65536)
SHIFT = 2


def shifted(row, dx, dy):
    out = [0] * 64
    for y in range(8):
        for x in range(8):
            sx, sy = x - dx, y - dy
            if 0 <= sx < 8 and 0 <= sy < 8:
                out[y * 8 + x] = row[sy * 8 + sx]
    return tuple(out)


def pool(n, seed):
    """n distinct 64-pixel images from the training rows (never dev or test), in one seeded shuffle."""
    tr = L.load(L.TASK["train"])
    train = tr[:len(tr) * 3 // 4]  # loop.py's dev split is tr[len*3//4:]; these are the rows before it
    seen, imgs = set(), []
    for r in train:
        for dy in range(-SHIFT, SHIFT + 1):
            for dx in range(-SHIFT, SHIFT + 1):
                im = shifted(r[:64], dx, dy)
                if im not in seen:
                    seen.add(im)
                    imgs.append(im)
    if len(imgs) < n:
        raise L.Failed(f"only {len(imgs)} distinct images can be made; asked for {n}")
    random.Random(seed).shuffle(imgs)
    return [list(im) for im in imgs[:n]]


def ols(xs, ys):
    """Least squares line through (x, y): (intercept, slope)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("need at least two distinct sizes")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def summarise(ts, sizes):
    """ts: {name: [ {N: ns} per round ]}. Fits per round; returns per-candidate and pairwise summaries."""
    out, slopes = {}, {}
    for name, rounds in ts.items():
        fits = [ols(sizes, [r[n] for n in sizes]) for r in rounds]
        a = [f[0] for f in fits]
        b = [f[1] for f in fits]
        slopes[name] = b
        resid = {n: [r[n] - (fa + fb * n) for r, (fa, fb) in zip(rounds, fits)] for n in sizes}
        rec = {"fixed_ns_per_run": L.mean_se(a), "ns_per_image": L.mean_se(b),
               "per_size": {str(n): {"ns_per_image": L.mean_se([r[n] / n for r in rounds]) if n else None,
                                     "run_ns": L.mean_se([r[n] for r in rounds]),
                                     "residual_ns": L.mean_se(resid[n])} for n in sizes}}
        if 1024 in sizes and 1025 in sizes:  # one image past a 1,024-image block: a block step shows here
            rec["step_1025_minus_1024_ns"] = L.mean_se([r[1025] - r[1024] for r in rounds])
        if 956 in sizes and 0 in sizes:  # loop.py's two task-size figures, from the same runs
            rec["task_meter_ns_per_image"] = L.mean_se([r[956] / 956 for r in rounds])
            rec["task_meter_marginal_ns_per_image"] = L.mean_se([(r[956] - r[0]) / 956 for r in rounds])
        out[name] = rec
    names = sorted(ts)
    pairs = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = [p - q for p, q in zip(slopes[x], slopes[y])]
            m, se = L.mean_se(d)
            pairs.append({"a": x, "b": y, "ns_per_image_diff": m, "ns_per_image_diff_se": se,
                          "ratio_of_means": statistics.fmean(slopes[x]) / statistics.fmean(slopes[y])})
    return out, pairs


def measure(dirs, sizes, rounds, seed):
    names = [os.path.basename(d.rstrip("/")) for d in dirs]
    if len(set(names)) != len(names):
        raise L.Failed(f"candidate names must be unique: {names}")
    imgs = pool(max(sizes), seed)
    rng = random.Random(seed)
    cpu = min(os.sched_getaffinity(0))
    work = tempfile.mkdtemp(prefix="exagium-scale-")
    problems = []
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
            c["ref"], _ = L.timed(c, imgs, timer, cpu)  # reference predictions on the whole pool (not timed into results)
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
    return {"tool": "loop/scale.py", "sizes": list(sizes), "rounds": rounds, "seed": seed,
            "pool": {"source": "first 3/4 of optdigits.tra, shifted -2..2 px, distinct", "n": max(sizes),
                     "sha256": L.sha256_bytes(bytes(v for im in imgs for v in im))},
            "host": {"cpu": L.cpu_model(), "jail": L.JAIL["mode"], "versions": L.versions()},
            "digests": L.digests(dirs) + [{"name": L.rel(os.path.abspath(__file__)), "sha256": L.sha256_file(os.path.abspath(__file__))}],
            "candidates": per, "pairs": pairs, "raw_ns": ts, "problems": problems}


def show(s):
    print(f"sizes {s['sizes']}, {s['rounds']} rounds, jail {s['host']['jail']}, {s['host']['cpu']}")
    for name, r in sorted(s["candidates"].items(), key=lambda kv: kv[1]["ns_per_image"][0]):
        b, bse = r["ns_per_image"]
        a, ase = r["fixed_ns_per_run"]
        tm = r.get("task_meter_marginal_ns_per_image")
        print(f"{name:16} per image {b:8.1f} ±{bse or 0:.1f} ns   fixed {a / 1e6:7.3f} ±{(ase or 0) / 1e6:.3f} ms"
              + (f"   task meter (N=956, minus t(0)) {tm[0]:7.1f} ±{tm[1] or 0:.1f}" if tm else ""))
        st = r.get("step_1025_minus_1024_ns")
        if st:
            print(f"                 t(1025) - t(1024): {st[0] / 1e3:+.1f} ±{(st[1] or 0) / 1e3:.1f} us "
                  f"(one image costs {b / 1e3:.2f} us on the line)")
        print("                 residual ns by size: " + ", ".join(
            f"{n}:{v['residual_ns'][0] / 1e3:+.0f}us" for n, v in r["per_size"].items()))
    for p in s["pairs"]:
        print(f"  {p['a']} - {p['b']}: {p['ns_per_image_diff']:+.1f} ±{p['ns_per_image_diff_se'] or 0:.1f} ns/img "
              f"(ratio {p['ratio_of_means']:.2f})")
    for x in s["problems"]:
        print("PROBLEM:", x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the result JSON here (never overwritten)")
    a = ap.parse_args(argv)
    try:
        sizes = sorted({int(x) for x in a.sizes.split(",")})
        if len(sizes) < 2 or sizes[0] < 0:
            ap.error("--sizes needs at least two distinct non-negative sizes")
        if a.rounds < 2:
            ap.error("--rounds must be >= 2")
        dirs = [L.candidate(d) for d in a.dirs]
        L.check_jail()
        s = measure(dirs, sizes, a.rounds, a.seed)
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
