"""Build the three task cards of the transfer plan (PLAN.md) as byte records.

python3 transfer/tasks.py OUT_DIR [--cache DIR]

Each task becomes OUT_DIR/<task>/ with:
  card.json   d (bytes per item), n_classes, per-feature type ("ordinal" | "categorical"), class names, sources
  train.x     n_train x d bytes      train.y   n_train bytes (class index)
  eval.x      n_eval x d bytes       eval.y    n_eval bytes
Every download is checked against the sha256 pinned below before use; a mismatch stops the build (exit 1).
The build is deterministic: the same inputs give the same bytes (eval items are a seeded sample, seed 1).
"""
import argparse, hashlib, json, os, sys, urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTDIGITS = os.path.join(HERE, "..", "loop", "task", "optdigits.tra")
SOURCES = {
    "optdigits.tra": ("file:loop/task/optdigits.tra",
                      "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"),
    "tinyshakespeare.txt": ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                            "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
                            "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"),
    "adult.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"),
    "adult.test": ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                   "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"),
}
TEXT_CONTEXT = 16       # bytes of context per next-character item
TEXT_TRAIN = 30000      # items drawn from the first 90% of the text
TEXT_EVAL = 5000        # items drawn from the last 10%
ADULT_NUMERIC = (0, 2, 4, 10, 11, 12)   # age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week


def fetch(name, cache):
    url, want = SOURCES[name]
    if url.startswith("file:"):
        path = os.path.join(HERE, "..", url[5:])
    else:
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            os.makedirs(cache, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(path + ".part", path)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        sys.exit(f"{name}: sha256 {got} != pinned {want}")
    return path


def write(out, name, card, trx, try_, evx, evy):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for fn, arr in (("train.x", trx), ("train.y", try_), ("eval.x", evx), ("eval.y", evy)):
        np.ascontiguousarray(arr, dtype=np.uint8).tofile(os.path.join(d, fn))
    card = dict(card, n_train=int(len(try_)), n_eval=int(len(evy)),
                files={fn: hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()
                       for fn in ("train.x", "train.y", "eval.x", "eval.y")})
    json.dump(card, open(os.path.join(d, "card.json"), "w"), indent=1, sort_keys=True)


def optdigits(out, cache):
    rows = np.loadtxt(fetch("optdigits.tra", cache), delimiter=",", dtype=np.int64)
    cut = len(rows) * 3 // 4   # loop.py's dev split: the last 25% of the training file
    X, y = rows[:, :64], rows[:, 64]
    write(out, "optdigits", dict(task="optdigits", d=64, n_classes=10, types=["ordinal"] * 64,
                                 classes=[str(i) for i in range(10)], kind="vision: 8x8 digit, pixel counts 0..16",
                                 split="train = first 75% of optdigits.tra, eval = last 25% (loop.py's dev split)"),
          X[:cut], y[:cut], X[cut:], y[cut:])


def text(out, cache):
    raw = np.frombuffer(open(fetch("tinyshakespeare.txt", cache), "rb").read(), dtype=np.uint8)
    cut = len(raw) * 9 // 10
    vocab = np.unique(raw[:cut])
    if not np.isin(raw[cut:], vocab).all():
        sys.exit("text: eval part has a byte not in the training vocabulary")
    cls = np.full(256, 255, dtype=np.int64)
    cls[vocab] = np.arange(len(vocab))
    rng = np.random.default_rng(1)

    def items(lo, hi, n):
        pos = np.sort(rng.choice(np.arange(lo + TEXT_CONTEXT, hi), size=n, replace=False))
        X = np.stack([raw[p - TEXT_CONTEXT:p] for p in pos])
        return X, cls[raw[pos]]

    trx, trY = items(0, cut, TEXT_TRAIN)
    evx, evY = items(cut, len(raw), TEXT_EVAL)
    write(out, "text", dict(task="text", d=TEXT_CONTEXT, n_classes=int(len(vocab)),
                            types=["categorical"] * TEXT_CONTEXT, classes=[chr(c) for c in vocab],
                            kind="language: next character from the previous 16 bytes (tinyshakespeare)",
                            split="train items: contexts ending in the first 90% of the text; eval: the last 10%"),
          trx, trY, evx, evY)


def adult(out, cache):
    def rows(path):
        out_ = []
        for line in open(path, encoding="ascii"):
            f = [s.strip() for s in line.strip().split(",")]
            if len(f) == 15:
                out_.append(f)
        return out_

    tr, ev = rows(fetch("adult.data", cache)), rows(fetch("adult.test", cache))
    label = lambda s: 1 if s.rstrip(".") == ">50K" else 0
    d = 14
    trX = np.zeros((len(tr), d), dtype=np.int64)
    evX = np.zeros((len(ev), d), dtype=np.int64)
    types = []
    for j in range(d):
        if j in ADULT_NUMERIC:
            types.append("ordinal")
            a = np.array([float(r[j]) for r in tr]); b = np.array([float(r[j]) for r in ev])
            edges = np.unique(np.quantile(a, np.arange(1, 255) / 255.0))   # at most 254 cut points -> bins 0..254
            trX[:, j] = np.searchsorted(edges, a, side="right"); evX[:, j] = np.searchsorted(edges, b, side="right")
        else:
            types.append("categorical")
            cats = sorted({r[j] for r in tr})
            idx = {c: i for i, c in enumerate(cats)}
            trX[:, j] = [idx[r[j]] for r in tr]
            evX[:, j] = [idx.get(r[j], 255) for r in ev]   # a category unseen in training -> 255
    write(out, "adult", dict(task="adult", d=d, n_classes=2, types=types, classes=["<=50K", ">50K"],
                             kind="tabular: UCI Adult census income; numeric columns as 255 training-quantile bins, "
                                  "categorical columns as the index of the category in training",
                             split="train = adult.data, eval = adult.test (the official split)"),
          trX, np.array([label(r[14]) for r in tr]), evX, np.array([label(r[14]) for r in ev]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cache", default=os.path.join(HERE, ".cache"))
    a = ap.parse_args()
    for f in (optdigits, text, adult):
        f(a.out, a.cache)
    print("built", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
