"""Build every task's splits as record files (runner/CONTRACT.md).

python3 tasks/build.py OUT_DIR [--cache DIR] [--only NAME ...]

Writes OUT_DIR/<task>/{train,dev,test}.{in,key} and OUT_DIR/<task>/manifest.json (sha256 of each file, item counts).
The label tasks reuse transfer/tasks.py's pinned downloads and byte encodings:
  optdigits: train = first 75% of optdigits.tra, dev = last 25% (loop.py's dev split), test = optdigits.tes;
  text, adult: train and dev = the first 80% and last 20% of transfer's training items, test = transfer's eval items
    (step 1 reported on them with nothing tuned; they are test from here on, behind the gate).
sat: planted random 3-SAT, generated here from fixed seeds; no download, no key (the checker verifies).
Deterministic: the same inputs give the same bytes; check_built() compares them with the cards' "data" digests.
"""
import argparse, hashlib, json, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "transfer"))
sys.path.insert(0, os.path.join(REPO, "runner"))
import records  # noqa: E402
import tasks as T  # noqa: E402  (transfer/tasks.py)

OPTDIGITS_TES = ("loop/task/optdigits.tes", "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8")
SAT = dict(n=50, ratio=4.2, k=3, seeds={"train": 101, "dev": 102, "test": 103},
           counts={"train": 100, "dev": 200, "test": 200})


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def save(out, name, splits):
    """splits: {split: (inputs list[bytes], keys list[bytes])}"""
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    man = {"task": name, "files": {}, "items": {}}
    for s, (ins, keys) in splits.items():
        assert len(ins) == len(keys)
        for ext, recs in (("in", ins), ("key", keys)):
            p = os.path.join(d, f"{s}.{ext}")
            records.write(p, recs)
            man["files"][f"{s}.{ext}"] = sha(p)
        man["items"][s] = len(ins)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1, sort_keys=True)


def from_transfer(tmp, name):
    d = os.path.join(tmp, name)
    card = json.load(open(os.path.join(d, "card.json")))
    rd = lambda f: np.fromfile(os.path.join(d, f), dtype=np.uint8)
    X, y = rd("train.x").reshape(-1, card["d"]), rd("train.y")
    EX, ey = rd("eval.x").reshape(-1, card["d"]), rd("eval.y")
    return X, y, EX, ey


def label_splits(X, y, cut, TX, ty):
    rec = lambda A, b: ([bytes(r) for r in A], [bytes([int(v)]) for v in b])
    return {"train": rec(X[:cut], y[:cut]), "dev": rec(X[cut:], y[cut:]), "test": rec(TX, ty)}


def build_label(out, cache, only):
    with tempfile.TemporaryDirectory() as tmp:
        if "optdigits" in only:
            T.optdigits(tmp, cache)
            X, y, EX, ey = from_transfer(tmp, "optdigits")   # transfer's train/eval = loop's train/dev
            path = os.path.join(REPO, OPTDIGITS_TES[0])
            if sha(path) != OPTDIGITS_TES[1]:
                sys.exit("optdigits.tes: sha256 does not match the pin")
            tes = np.loadtxt(path, delimiter=",", dtype=np.int64)
            AX, Ay = np.concatenate([X, EX]), np.concatenate([y, ey])
            save(out, "optdigits", label_splits(AX, Ay, len(X), tes[:, :64].astype(np.uint8), tes[:, 64]))
        for name, fn in (("text", T.text), ("adult", T.adult)):
            if name in only:
                fn(tmp, cache)
                X, y, EX, ey = from_transfer(tmp, name)
                save(out, name, label_splits(X, y, len(X) * 4 // 5, EX, ey))


def planted_3sat(rng, n, m):
    """One satisfiable instance: a hidden assignment, and m clauses of 3 distinct variables it satisfies."""
    truth = rng.integers(0, 2, n)
    clauses = []
    while len(clauses) < m:
        v = rng.choice(n, 3, replace=False)
        s = rng.integers(0, 2, 3)          # 1 = positive literal
        if any((truth[v[i]] == 1) == (s[i] == 1) for i in range(3)):
            clauses.append([int(v[i] + 1) * (1 if s[i] else -1) for i in range(3)])
    lits = np.array(clauses, dtype=np.int8).ravel()
    return bytes([n, m % 256, m // 256]) + lits.tobytes()


def build_sat(out):
    n, m = SAT["n"], int(round(SAT["n"] * SAT["ratio"]))
    splits = {}
    for s in ("train", "dev", "test"):
        rng = np.random.default_rng(SAT["seeds"][s])
        ins = [planted_3sat(rng, n, m) for _ in range(SAT["counts"][s])]
        splits[s] = (ins, [b""] * len(ins))
    save(out, "sat", splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cache", default=os.path.join(REPO, "transfer", ".cache"))
    ap.add_argument("--only", nargs="*", default=["optdigits", "text", "adult", "sat"])
    a = ap.parse_args()
    build_label(a.out, a.cache, set(a.only))
    if "sat" in a.only:
        build_sat(a.out)
    bad = []
    for name in a.only:
        man = json.load(open(os.path.join(a.out, name, "manifest.json")))
        card = json.load(open(os.path.join(HERE, name, "card.json")))
        if card.get("data") and card["data"] != man["files"]:
            bad.append(name)
        print(name, man["items"])
    if bad:
        sys.exit(f"built data differs from the card's pinned digests: {bad}")


if __name__ == "__main__":
    main()
