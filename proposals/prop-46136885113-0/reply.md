```hypothesis
Family: label-tinymlp (tiny MLP with 1 hidden layer of 4-8 ReLU units).
Why: Adult has 14 mixed-type columns; label-linear (0.8524) underfits interactions, label-mlp16 (0.8591) over-parameters (16 units → 2325 Ir/item). A tiny MLP (4-8 units) should capture key non-linearities with far fewer ops, closing the accuracy gap at lower cost. Tuning H on the last 20% of train.in selects the sweet spot.
Failure signal: dev accuracy ≤ 0.8524 or Ir/item ≥ 1500 (no improvement over linear or too costly).
```

```python
"""label-tinymlp: tiny MLP (1 hidden layer, 4-8 ReLU units) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses H on tuning split (last 20%),
then trains on all data. Exports same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [4, 8]  # tiny hidden sizes


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Fit per-column encoders on full training data."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode_all(X, enc):
    """Apply encoders to data matrix."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_mlp(Z, y, H, C, rng, epochs):
    """Train MLP with Adam, return params (W1,b1,W2,b2)."""
    n, k = Z.shape
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            gh = g @ W2.T
            gh[a <= 0] = 0.0
            grads = [
                z.T @ gh + L2 * W1,
                gh.sum(0),
                h.T @ g + L2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def eval_acc(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def export_weights(card, enc, params, path):
    """Write weights.bin in label-linear format."""
    d = card["input"]["bytes"]
    H = params[0].shape[1]
    C = card["classes"]
    W = H
    # Build per-column lookup tables T[j][v][w] = enc[j][v] @ W1[col_slice]
    off, T = 0, []
    for E in enc:
        T.append((E @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()      # b1
    out += params[2].astype("<f4").tobytes()      # W2
    out += params[3].astype("<f4").tobytes()      # b2
    open(path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(X)
    tune_size = n // 5
    train_idx = slice(0, n - tune_size)
    tune_idx = slice(n - tune_size, n)

    # Fit encoders on ALL training data (as card specifies)
    enc = build_encoders(X, card["input"]["types"])
    Z = encode_all(X, enc)

    # Grid search H on tuning split
    best_H, best_acc = H_CHOICES[0], -1.0
    for H in H_CHOICES:
        rng = np.random.default_rng(SEED + H * 1000)  # deterministic per H
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / (n - tune_size)))
        params = train_mlp(Z[train_idx], y[train_idx], H, C, rng, epochs)
        acc = eval_acc(Z[tune_idx], y[tune_idx], params)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on full data with best H
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_mlp(Z, y, best_H, C, rng, epochs)
    export_weights(card, enc, params, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* label-tinymlp infer: identical weight format to label-linear (supports H>0).
 * ./infer weights.bin in.rec N  -> writes N single-byte class records to stdout. */
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