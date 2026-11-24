```hypothesis
Family: label-ngram (discriminative linear model on hashed suffix n-grams orders 1–5).
Why: Character-level next-char prediction benefits from local sequential dependencies (suffix n-grams) that position-wise unigrams (label-linear) miss. Hashing n-grams into modest tables (256..4096 buckets) adds only ~3.5× the parameters of label-linear but captures variable-order Markov structure. Trained with the same Adam/L2 regimen on the tuning split (last 20% of train.in) for hash-size selection, then retrained on all train, it should exceed 0.35 dev accuracy at ~2× label-linear's inference cost.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear's score) or inference cost > 3× label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

ORDERS = [1, 2, 3, 4, 5]
HASH_BITS = [8, 11, 12, 12, 12]
HASH_SIZES = [1 << b for b in HASH_BITS]
HASH_MASKS = [(1 << b) - 1 for b in HASH_BITS]
FNV_SEED = 0x9e3779b9
FNV_PRIME = 0x1000193

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_indices_vec(x, order_idx):
    o = ORDERS[order_idx]
    mask = HASH_MASKS[order_idx]
    if o == 1:
        return x[:, -1].astype(np.uint32)
    h = np.full(x.shape[0], FNV_SEED, dtype=np.uint64)
    for i in range(o):
        h = (h ^ x[:, -o + i].astype(np.uint64)) * FNV_PRIME
    return (h & mask).astype(np.int32)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = n * 4 // 5
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:],