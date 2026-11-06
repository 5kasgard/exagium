#!/usr/bin/env python3
"""Z's independent adversarial test of probe.describe_levels (v2, fixed checker).

My own synthetic generator: gamma noise, correlated shocks, off-grid capacities,
curved ramps, interference spikes, soft (hit-rate) mixtures, per-rep capacity jitter.
200 seeds per case, calling describe_levels directly.

Errors counted:
  phantom    - more levels fitted than truth, or two fitted levels on one true level (split)
  false-bd   - a fitted level whose value maps to no true latency (log-dist > ln 1.5),
               or whose span midpoint lies inside a true transition zone (ramp/soft cases)
  wrong-br   - a hard-step capacity missed by its level's bracket (bracket ends >1 grid
               step below, or starts >1 grid step above the true capacity)
  miss       - a true level with no fitted level on it
Tool passes a case when phantom and false-bd stay <= 5% (VERIFY.md claim 4's bar).
"""
import math, random, sys
sys.path.insert(0, "/home/z/my-project/exagium/probe")
import probe

GRID = sorted(set(2 ** e * m for e in range(9, 29) for m in (1, 1.5, 2, 3, 4, 6, 8)))
GRID = [g for g in GRID if g <= 2 << 30]
LAT = (1.4, 4.5, 18.0, 90.0)
CAPS = (48 * 1024, int(2.2 * 1024 * 1024), 380 * 1024 * 1024)

def synth(seed, *, reps=7, cv=0.02, ar=0.35, spikes=0.0, spike_x=3.0, soft=False,
          cap_jitter=0.0, ramped=False):
    rng = random.Random(seed)
    rep_caps = [[c * (1 + cap_jitter * (2 * rng.random() - 1)) for c in CAPS] for _ in range(reps)]
    meds, ses, rmax, rmin = [], [], [], []
    prev = 0.0
    for s in GRID:
        per_rep = []
        for r in range(reps):
            v = LAT[-1]
            for i, c in enumerate(rep_caps[r]):
                if s <= c:
                    v = LAT[i]
                    break
            if soft:
                for i, c in enumerate(rep_caps[r]):
                    if s <= c:
                        break
                    if s < c * 4:
                        hr = min(1.0, c / s)
                        v = LAT[i] * hr + LAT[min(i + 1, len(LAT) - 1)] * (1 - hr)
                        break
            elif ramped:
                for i, c in enumerate(rep_caps[r]):
                    if c < s < c * 1.25:
                        f = math.log(s / c) / math.log(1.25)
                        v = LAT[i] * (1 - f) + LAT[i + 1] * f
                        break
            per_rep.append(v)
        if rng.random() < spikes:
            per_rep = [v * spike_x for v in per_rep]
        noisy = []
        for v in per_rep:
            g = rng.gammavariate(4.0, 0.25)          # mean 1, skew
            eps = cv * (g - 1.0) + ar * prev
            noisy.append(v * (1 + eps))
            prev = eps
        m = sum(noisy) / reps
        sd = (sum((x - m) ** 2 for x in noisy) / (reps - 1)) ** 0.5
        meds.append(m); ses.append(sd / reps ** 0.5)
        rmax.append(max(noisy)); rmin.append(min(noisy))
    return meds, ses, rmax, rmin

def gidx(b):
    return min(range(len(GRID)), key=lambda i: abs(math.log(GRID[i]) - math.log(b)))

def check(seed, **kw):
    meds, ses, rmax, rmin = synth(seed, **kw)
    reps = kw.get("reps", 7)
    out = probe.describe_levels(GRID, meds, ses, df=reps - 1, rep_max=rmax, rep_min=rmin)
    if out is None:
        return "refused"
    levels = [l for l in out if l["kind"] == "level"]
    errs = set()
    # map each fitted level to a true level by closest log-latency
    taken = []
    for l in levels:
        v = l["latency_ns"]
        li = min(range(len(LAT)), key=lambda i: abs(math.log(v) - math.log(LAT[i])))
        if abs(math.log(v) - math.log(LAT[li])) > math.log(1.5):
            errs.add("false-bd")
            continue
        if taken.count(li):
            errs.add("phantom")          # two levels on one true level = split
        taken.append(li)
        mid = math.sqrt(l["from_bytes"] * l["to_bytes"])
        if kw.get("soft") or kw.get("ramped"):
            for c in CAPS:
                if c < mid < c * 1.25:
                    errs.add("false-bd")  # a "level" sitting on a true transition
                    break
    if len(levels) > len(LAT):
        errs.add("phantom")
    if len(set(taken)) < len(LAT):
        errs.add("miss")
    # brackets: each capacity must be inside its owning level's bracket (the level whose
    # span lies below c), allowing a null upper end when no next level was found
    if not (kw.get("soft") or kw.get("ramped")):
        for c in CAPS[:2]:
            owner = None
            for l in levels:
                if l["from_bytes"] < c and l["to_bytes"] < c * 1.6:
                    if owner is None or l["to_bytes"] > owner["to_bytes"]:
                        owner = l
            if owner is None:
                continue  # no level below c at all: counted as miss already
            br = owner.get("capacity_bytes_between")
            if not br or br[0] is None:
                continue
            ok = br[0] <= c * 1.02 and (br[1] is None or br[1] >= c * 0.99)
            if not ok:
                errs.add("wrong-br")
    return "+".join(sorted(errs)) if errs else "ok"

def run():
    CASES = {
        "clean 7reps cv2%": dict(),
        "noisy cv5% ar0.5": dict(cv=0.05, ar=0.5),
        "heavy tail cv4%": dict(cv=0.04, ar=0.2, spikes=0.04, spike_x=4.0),
        "spikes 10% x3": dict(spikes=0.10, spike_x=3.0, cv=0.03),
        "soft transitions": dict(soft=True, cv=0.03),
        "cap jitter 20% 3reps": dict(cap_jitter=0.20, reps=3, cv=0.03),
        "cap jitter 30% 7reps": dict(cap_jitter=0.30, reps=7, cv=0.03),
        "curved ramps": dict(ramped=True, cv=0.015),
        "ramps+spikes+cv5": dict(ramped=True, spikes=0.08, spike_x=2.5, cv=0.05, ar=0.4),
    }
    N = 200
    print(f"{'case':26s} {'ok':>4s} {'phantom':>8s} {'falsebd':>8s} {'wrongbr':>8s} {'miss':>5s} {'refus':>5s}")
    worst = 0.0
    for name, kw in CASES.items():
        counts = {}
        for seed in range(N):
            r = check(seed, **kw)
            counts[r] = counts.get(r, 0) + 1
        ph = sum(v for k, v in counts.items() if "phantom" in k)
        fb = sum(v for k, v in counts.items() if "false-bd" in k)
        wb = sum(v for k, v in counts.items() if "wrong-br" in k)
        ms = sum(v for k, v in counts.items() if "miss" in k)
        rf = counts.get("refused", 0)
        ok = counts.get("ok", 0)
        print(f"{name:26s} {ok:4d} {100*ph/N:7.1f}% {100*fb/N:7.1f}% {100*wb/N:7.1f}% {100*ms/N:4.0f}% {100*rf/N:4.0f}%")
        worst = max(worst, 100 * ph / N, 100 * fb / N)
    print(f"\nworst phantom/false-boundary rate: {worst:.1f}%  (bar: 5%)")

if __name__ == "__main__":
    run()
