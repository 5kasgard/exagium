You write a new candidate model for a public research repository that measures small machine-learning
models on accuracy and CPU cost. Your code is compiled and run in a sandbox with no network; it is judged on a held-back
dev split by a fixed checker.

# The contract
## A candidate (`runner/candidates/<name>/`): code only, no weights
- `train.py`: `python3 -s -P train.py card.json train.in train.key weights.bin` (stdlib + numpy, deterministic,
  one process, single thread). A candidate with nothing to learn still writes a weights file.
- `infer.c`: `./infer weights.bin in.rec N` writes exactly N output records to stdout. N = 0 is start-up only.
- Weights are made by training in the jail at run time. `reproduce` trains twice and requires identical bytes;
  the test gate requires it.


Rules that the checks enforce:
- train.py: Python 3 with the standard library and numpy only; deterministic (fixed seeds); single thread; it must finish
  within 20 minutes on one CPU core and within 8 GiB of memory; it reads the card and training records and writes
  weights.bin. Running it twice must give byte-identical weights.bin.
- infer.c: C11, libc and libm only, compiled with gcc -O2 -std=c11 -ffp-contract=off; `./infer weights.bin in.rec N`
  reads N input records and writes exactly N output records to stdout (4-byte little-endian length, then the bytes).
  For this task an output record is one byte: the class index. Validate the weights file size; never read past buffers.
- If you choose settings, choose them on the card's tuning split only (last 20% of train.in, file order), then train
  on all of train.

# The task card
```json
{
 "task": "optdigits",
 "kind": "vision",
 "description": "8x8 handwritten digit, pixel counts 0..16 (UCI optdigits)",
 "input": {
  "bytes": 64,
  "meaning": "64 pixel counts, row-major",
  "types": [
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal",
   "ordinal"
  ]
 },
 "output": {
  "max_len": 1,
  "meaning": "the digit, one byte 0..9"
 },
 "classes": 10,
 "checker": "label",
 "metric": "accuracy",
 "baselines": [
  "majority",
  "label-linear"
 ],
 "floor": "majority",
 "sources": [
  "loop/task/optdigits.tra",
  "loop/task/optdigits.tes"
 ],
 "splits": {
  "train": "train.in + train.key, seen by train.py",
  "dev": "scored by the runner, never trained on",
  "test": "scored only under a ledger plan (test_gate)"
 },
 "test_gate": "plan",
 "items": {
  "dev": 956,
  "test": 1797,
  "train": 2867
 },
 "data": {
  "dev.in": "78cb337315bc25c685218580b4dd22f555a1b6354256c89dd5453be88da6d483",
  "dev.key": "c4d6a7b1f2847cc256e2cdc6647ccf05f5634cb47173eddbdc37265814f458a5",
  "test.in": "68f8bc193c78678b33fd19fa8a766268f1b9d9307e2246ad04321bdfe4a20ab1",
  "test.key": "200fd39ba30e931a8595a5a1d0162d88cba8139d57f53842525a4406629f89ac",
  "train.in": "28ffad9620e10ac48d4e2cffe12ca482b644140198514f99da9d8ee9ae2a7e0d",
  "train.key": "0a62505a925d91b0a1213c7043946547802f940d419ad023fba92a810e4f506f"
 },
 "tuning": "the last 20% of train.in in file order; a candidate that chooses settings chooses them there, then trains on all of train (queue P3 step 1)"
}

```

# A working example candidate (label-linear): follow its input and output handling exactly
```python
"""Table family for any label task whose items are fixed-size bytes (ported from transfer/families.py).
python3 -s -P train.py card.json train.in train.key weights.bin   (deterministic: seed 1, single thread)
Encoding by the card's per-byte types: ordinal z-scored, categorical one-hot over training values. H hidden ReLU
units (0 = logistic regression). Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000 / n)).
Exported as per-byte lookup tables: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H:
float32 W2[H][C], b2[C] (W = H, or C for linear)."""
import json, math, struct, sys

import numpy as np

H = 0
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32); b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32); b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32); b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True); P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1; h = np.maximum(a, 0); g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32)); off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes() + params[3].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()

```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

/* next record of a record file: 4-byte little-endian length, then the bytes. */
static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}

```

# Families that already exist (propose a different one)
- label-gbdt: Gradient-boosted trees for label tasks with fixed-size byte items (queue P3 step 1; outside-strength baseline).
- label-knn: k nearest neighbours for any label task with fixed-size byte items (queue P3 step 1).
- label-linear: Table family for any label task whose items are fixed-size bytes (ported from transfer/families.py).
- label-mlp16: Table family for any label task whose items are fixed-size bytes (ported from transfer/families.py).
- label-mlp2l: (no description in train.py)
- label-wisard: WiSARD n-tuple classifier (weightless: lookup tables only) for label tasks with fixed-size byte items (queue P3 step
- majority: Majority baseline for any label task: weights.bin is the most frequent training key (ties: the smallest bytes).

# Current dev results on this task (accuracy, instructions per item, ns per item)
optdigits / dev (956 items, accuracy; floor majority = 0.0931; jail bwrap)
  label-gbdt     0.9791 ± 0.0046  Ir/item    41742.7  ns/item    4546.8  above floor: True
  label-mlp16    0.9770 ± 0.0049  Ir/item     8685.5  ns/item     776.4  above floor: True
  label-knn      0.9718 ± 0.0054  Ir/item  2237451.6  ns/item  136816.7  above floor: True
  label-linear   0.9613 ± 0.0062  Ir/item     5058.6  ns/item     416.9  above floor: True
  label-wisard   0.9582 ± 0.0065  Ir/item    34076.3  ns/item    2867.4  above floor: True
  majority       0.0931 ± 0.0094  Ir/item      322.0  ns/item      20.5  above floor: None

# Your answer
Propose ONE new model family for this task that you expect to beat label-linear on dev accuracy, or to match it at a
lower cost per item. Answer with exactly three fenced blocks, in this order and nothing else:
```hypothesis
(at most 10 lines: the family, why it should do well here, and what result would show it does not)
```
```python
(the complete train.py)
```
```c
(the complete infer.c)
```
