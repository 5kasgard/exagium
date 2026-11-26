"""Can the drift task (tasks/drift, queue K88) show a stateful learner's advantage at all? (queue P3, K89)"""
import importlib.util, json, os, random, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMPY = "numpy==2.5.3"   # the version runner.yml pins; the card's digests were made with it (PR #244)

PLAN = ("Data: tasks/build.py build_drift (seed 2026, 8000 rows, 16 features, the rule changes every 2000 rows; train = "
        "blocks 0-1, dev = block 2, test = block 3), checked against tasks/drift/card.json's pinned digests. Static: "
        "logistic regression trained on train only (full-batch gradient descent, 300 steps, step 0.5, no penalty), then "
        "frozen. Online: the same model, then run through dev and test in stream order: predict each row, then take one "
        "SGD step (step 0.05) on its label (prequential). Score: accuracy on the 2000 test rows. Majority: the most common "
        "train label, scored on test. Rule, fixed before the run: holds if online minus static is at least 0.10, the "
        "2.5th percentile of 2000 paired bootstrap resamples of that difference (seed 0) is above 0.05, and online beats "
        "majority by at least 0.10. Otherwise it fails. Invalid if the built data does not match the card. Margin: 0.10 is "
        "chosen to mean an advantage big enough to rank families, not just one outside noise. One proportion over 2000 "
        "items has a standard error of at most 0.011. If it holds, the drift task stays in P3's suite as the stateful "
        "task, and the runner contract needs a labelled-feedback mode. If it fails, the task's drift is too weak or too "
        "strong to tell the families apart, and its parameters are changed under a new rule. Forgetting is reported but "
        "not judged: block-0 accuracy of each model after the stream.")


def numpy():
    try:
        import numpy as np
        return np
    except ImportError:   # the experiment job's runner has no numpy (experiment.yml installs nothing); a pinned private copy
        d = tempfile.mkdtemp()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", d, "--break-system-packages", NUMPY], check=True)
        sys.path.insert(0, d)
        import numpy as np
        return np


def data():
    np = numpy()
    spec = importlib.util.spec_from_file_location("drift_build", os.path.join(ROOT, "tasks", "build.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    card = json.load(open(os.path.join(ROOT, "tasks", "drift", "card.json")))
    with tempfile.TemporaryDirectory() as out:
        B.build_drift(out)
        man = json.load(open(os.path.join(out, "drift", "manifest.json")))
    X, y, _ = B.drift.generate(B.DRIFT["seed"], B.DRIFT["n"], B.DRIFT["dim"], B.DRIFT["segments"])
    Xf = np.hstack([X.astype(np.float64) / 255.0 * 8.0 - 4.0, np.ones((len(X), 1))])
    return np, Xf, y.astype(np.float64), man["files"] == card.get("data"), np.__version__


def run():
    np, X, y, pinned, ver = data()
    b = len(X) // 4
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    w = np.zeros(X.shape[1])
    for _ in range(300):
        w -= 0.5 * X[:2 * b].T @ (sig(X[:2 * b] @ w) - y[:2 * b]) / (2 * b)
    static = w.copy()
    right_static = ((X[3 * b:] @ static > 0) == (y[3 * b:] == 1)).astype(int)
    right_online = []
    for t in range(2 * b, 4 * b):
        if t >= 3 * b:
            right_online.append(int((X[t] @ w > 0) == (y[t] == 1)))
        w -= 0.05 * (sig(X[t] @ w) - y[t]) * X[t]
    right_online = np.array(right_online)
    maj = float(y[:2 * b].mean() >= 0.5)
    majority = float((y[3 * b:] == maj).mean())
    rng, n, diffs = random.Random(0), len(right_online), []
    d = right_online - right_static
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(d[i] for i in idx) / n)
    diffs.sort()
    acc0 = lambda v: float(((X[:b] @ v > 0) == (y[:b] == 1)).mean())
    return {"pinned": bool(pinned), "numpy": ver, "test_items": n, "static": round(float(right_static.mean()), 4),
            "online": round(float(right_online.mean()), 4), "majority": round(majority, 4),
            "diff": round(float(d.mean()), 4), "diff_p2_5": round(diffs[int(0.025 * len(diffs))], 4),
            "forgetting_block0": {"static": round(acc0(static), 4), "online_after_stream": round(acc0(w), 4)}}


def rule(r):
    if not r.get("pinned") or r.get("test_items") != 2000:
        return {"verdict": "invalid", "why": f"built data does not match the card (pinned {r.get('pinned')}, test {r.get('test_items')})"}
    ok = r["diff"] >= 0.10 and r["diff_p2_5"] > 0.05 and r["online"] - r["majority"] >= 0.10
    return {"verdict": "holds" if ok else "fails",
            "why": f"online {r['online']} vs static {r['static']} (diff {r['diff']}, p2.5 {r['diff_p2_5']}), majority {r['majority']}"}
