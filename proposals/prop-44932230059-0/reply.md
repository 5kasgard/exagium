```hypothesis
Family: label-ada (AdaBoost with decision stumps on encoded features).  
Why: The Adult dataset has mixed ordinal/categorical features; stumps on one-hot and z-scored inputs capture non-linear interactions missed by logistic regression, while staying extremely fast at inference (O(T·d) with T≈100).  
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear by >2×.
```

```python
"""label-ada: AdaBoost with decision stumps on encoded features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encoding: ordinal -> z-scored, categorical -> one-hot over training values (255=unseen -> zero).
Tunes number of stumps on last 20% of train.in, then retrains on all data.
Weights format: magic 0x41444131, d, T, mu[d], sd[d], then T stumps (col u8, type u8, threshold f32, polarity i8, alpha f32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
T_CANDIDATES = [50, 100, 150, 200, 250]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(card, X):
    """Return Z (n, k), and metadata: for each encoded feature, (col, type, category_or_None)."""
    enc = []
    meta = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            scaled = ((np.arange(256, dtype=np.float64) - mu) / sd).astype(np.float32)
            enc.append(scaled[v])
            meta.append((j, 0, None, mu, sd))
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E[v])
            for idx, cat in enumerate(vals):
                meta.append((j, 1, cat, 0.0, 1.0))
    Z = np.column_stack(enc).astype(np.float32)
    return Z, meta


def train_ada(Z, y, T, rng):
    n, k = Z.shape
    w = np.ones(n, dtype=np.float32) / n
    y_bin = (y * 2 - 1).astype(np.float32)  # -1, +1
    stumps = []  # each: (feat_idx, threshold, polarity, alpha)
    for _ in range(T):
        best_err = 1.0
        best_feat = 0
        best_thr = 0.0
        best_pol = 1
        for f in range(k):
            vals = Z[:, f]
            uniq = np.unique(vals)
            if len(uniq) == 1:
                continue
            for thr in (uniq[:-1] + uniq[1:]) / 2.0:
                for pol in (1, -1):
                    pred = np.where(vals > thr, pol, -pol).astype(np.float32)
                    err = np.sum(w * (pred != y_bin))
                    if err < best_err:
                        best_err = err
                        best_feat = f
                        best_thr = thr
                        best_pol = pol
        if best_err >= 0.5 or best_err == 0.0:
            break
        alpha = 0.5 * math.log((1.0 - best_err) / best_err)
        pred = np.where(Z[:, best_feat] > best_thr, best_pol, -best_pol).astype(np.float32)
        w *= np.exp(-alpha * y_bin * pred)
        w /= w.sum()
        stumps.append((best_feat, best_thr, best_pol, alpha))
    return stumps


def predict_ada(Z, stumps):
    scores = np.zeros(Z.shape[0], dtype=np.float32)
    for feat, thr, pol, alpha in stumps:
        pred = np.where(Z[:, feat] > thr, pol, -pol).astype(np.float32)
        scores += alpha * pred
    return (scores > 0).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    X_train, y_train = X[:-tune_size], y[:-tune_size]
    X_tune, y_tune = X[-tune_size:], y[-tune_size:]

    Z_train, meta = encode(card, X_train)
    Z_tune, _ = encode(card, X_tune)

    rng = np.random.default_rng(SEED)
    best_T, best_acc = T_CANDIDATES[0], -1.0
    for T in T_CANDIDATES:
        stumps = train_ada(Z_train, y_train, T, rng)
        pred = predict_ada(Z_tune, stumps)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_T = acc, T

    Z_all, meta = encode(card, X)
    stumps = train_ada(Z_all, y, best_T, rng)

    d = card["input"]["bytes"]
    mu = np.zeros(d, dtype=np.float32)
    sd = np.ones(d, dtype=np.float32)
    for col, typ, cat, m, s in meta:
        if typ == 0:
            mu[col], sd[col] = m, s

    out = struct.pack("<3I", 0x41444131, d, len(stumps))
    out += mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    for feat, thr, pol, alpha in stumps:
        col, typ, cat, _, _ = meta[feat]
        threshold = thr if typ == 0 else float(cat)
        out += struct.pack("<BBfbf", col, typ, threshold, pol, alpha)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-ada inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output record: 1 byte (class 0/1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x41444131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], T = hdr[2];
    float *mu = (float *)(hdr + 3);
    float *sd = mu + d;
    unsigned char *sp = (unsigned char *)(sd + d);
    long expected = 12L + 8L * d + 11L * T;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = 0.0f;
        for (int t = 0; t < T; t++) {
            int col = sp[0];
            int type = sp[1];
            float thr; memcpy(&thr, sp + 2, 4);
            int8_t pol; memcpy(&pol, sp + 6, 1);
            float alpha; memcpy(&alpha, sp + 7, 4);
            sp += 11;
            unsigned char byte = xi[col];
            int pred;
            if (type == 0) {
                float val = (byte - mu[col]) / sd[col];
                pred = (val > thr) ? 1 : -1;
            } else {
                pred = (byte == (unsigned char)thr) ? 1 : -1;
            }
            score += alpha * pol * pred;
        }
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```