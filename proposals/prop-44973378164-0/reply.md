```hypothesis
Factorization Machine (label-fm) with 8-d embeddings captures pairwise interactions between the 14 mixed ordinal/categorical features. This suits Adult's known feature synergies (e.g., education×occupation) better than linear models while staying faster than GBDT/MLP. If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 719, the family does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
GRID = {
    "k": [4, 8],
    "lr": [0.01, 0.03],
    "l2": [0.0, 1e-4],
    "epochs": [30],
    "batch": [256],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, w0, w, V, k):
    n, d = X.shape
    scores = np.full(n, w0, dtype=np.float32)
    for j in range(d):
        idx = X[:, j]
        scores += w[j, idx]
        v = V[j, idx]                     # (n, k)
        sum_v = v.sum(axis=1)             # (n,)
        sum_v2 = (v * v).sum(axis=1)
        scores += 0.5 * (sum_v * sum_v - sum_v2)
    return scores

def fm_backward(X, y, scores, w, V, k, l2):
    n, d = X.shape
    # logistic loss gradient: dL/ds = -y / (1 + exp(y*s))  with y in {-1,1}
    y_pm = 2.0 * y - 1.0
    margin = y_pm * scores
    grad_s = -y_pm / (1.0 + np.exp(margin))   # (n,)
    gw0 = grad_s.mean()
    gw = np.zeros_like(w)
    gV = np.zeros_like(V)
    for j in range(d):
        idx = X[:, j]
        np.add.at(gw[j], idx, grad_s)
        v = V[j, idx]                         # (n, k)
        sum_v = v.sum(axis=0)                 # (k,)
        # d interaction / d v_j[f] = sum_v[f] - v_j[f]
        gv = grad_s[:, None] * (sum_v - v)    # (n, k)
        np.add.at(gV[j], idx, gv)
    if l2:
        gw += l2 * w
        gV += l2 * V
    return gw0, gw, gV

def train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, batch, rng):
    n, d = X_tr.shape
    w0 = 0.0
    w = np.zeros((d, 256), np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb, yb = X_tr[idx], y_tr[idx]
            scores = fm_forward(Xb, w0, w, V, k)
            gw0, gw, gV = fm_backward(Xb, yb, scores, w, V, k, l2)
            t += 1
            b1, b2 = 0.9, 0.999
            m_w0 = b1 * m_w0 + (1 - b1) * gw0
            v_w0 = b2 * v_w0 + (1 - b2) * gw0 * gw0
            w0 -= lr * (m_w0 / (1 - b1**t)) / (np.sqrt(v_w0 / (1 - b2**t)) + 1e-8)
            m_w = b1 * m_w + (1 - b1) * gw
            v_w = b2 * v_w + (1 - b2) * gw * gw
            w -= lr * (m_w / (1 - b1**t)) / (np.sqrt(v_w / (1 - b2**t)) + 1e-8)
            m_V = b1 * m_V + (1 - b1) * gV
            v_V = b2 * v_V + (1 - b2) * gV * gV
            V -= lr * (m_V / (1 - b1**t)) / (np.sqrt(v_V / (1 - b2**t)) + 1e-8)
    val_scores = fm_forward(X_val, w0, w, V, k)
    val_pred = (val_scores > 0).astype(np.int64)
    return (val_pred == y_val).mean(), (w0, w, V)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc, best_cfg, best_params = -1, None, None
    rng = np.random.default_rng(SEED)
    for k in GRID["k"]:
        for lr in GRID["lr"]:
            for l2 in GRID["l2"]:
                for epochs in GRID["epochs"]:
                    for batch in GRID["batch"]:
                        acc, params = train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, batch, rng)
                        if acc > best_acc:
                            best_acc, best_cfg, best_params = acc, (k, lr, l2, epochs, batch), params
    # Retrain on full data with best config
    k, lr, l2, epochs, batch = best_cfg
    rng_full = np.random.default_rng(SEED)
    _, final_params = train_fm(X, y, X[:1], y[:1], k, lr, l2, epochs, batch, rng_full)
    w0, w, V = final_params

    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, k, 2)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || k > 32 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = w0;
        float sum_v[32] = {0}, sum_v2[32] = {0};
        for (int j = 0; j < d; j++) {
            int idx = xi[j];
            score += w[j * 256 + idx];
            const float *v = V + (j * 256 + idx) * k;
            for (int f = 0; f < k; f++) {
                float vf = v[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
            }
        }
        for (int f = 0; f < k; f++) score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```