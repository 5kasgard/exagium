```hypothesis
Family: label-embed-mlp. Uses per-position byte embeddings (dimension D) concatenated, one ReLU hidden layer (size H), then linear to 65 classes. Precomputes per-position lookup tables for the first layer so inference is two matrix-vector products (embedding sum + ReLU + output). Grid-searches (D,H) on the tuning split (last 20% of train) then retrains on all data. Expected to capture positional interactions better than linear at similar or lower CPU cost (H=64 gives ~12k FLOPs vs linear's 266k). Failure: dev accuracy ≤ 0.3375 or CPU cost ≥ label-linear.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Grid-searches (D,H) on tuning split (last 20% of train), then trains on all.
Exports: magic 0x4D4C5031, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
GRID_EPOCHS, FINAL_EPOCHS = 10, 50
D_CHOICES = [4, 8, 16]
H_CHOICES = [32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of CE wrt logits (n, C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return (probs / n).astype(np.float32)

def train_model(X, y, D, H, epochs, rng):
    n, d = X.shape
    C = 65
    # embeddings: list of d arrays (256, D)
    E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    emb_params = E
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    emb_state = [(np.zeros_like(e), np.zeros_like(e)) for e in emb_params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            emb = np.concatenate([E[j][xb[:, j]] for j in range(d)], axis=1)  # (b, d*D)
            h_pre = emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            # loss grad
            g = softmax_cross_entropy_grad(logits, yb)
            # backward
            dW2 = h.T @ g + L2 * W2
            db2 = g.sum(0)
            dh = (g @ W2.T) * (h_pre > 0)
            dW1 = emb.T @ dh + L2 * W1
            db1 = dh.sum(0)
            demb = dh @ W1.T  # (b, d*D)
            # Adam updates
            grads = [dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            # embedding updates
            off = 0
            for j in range(d):
                ej = E[j]
                gr_j = demb[:, off:off + D]
                off += D
                # accumulate gradient per embedding index
                # we have xb[:, j] as indices (b,), gr_j as (b, D)
                # use np.add.at for deterministic unordered accumulation
                m_e, v_e = emb_state[j]
                grad_e = np.zeros_like(ej)
                np.add.at(grad_e, xb[:, j], gr_j)
                grad_e += L2 * ej
                m_e *= 0.9; m_e += 0.1 * grad_e; v_e *= 0.999; v_e += 0.001 * grad_e * grad_e
                ej -= LR * (m_e / (1 - 0.9 ** t)) / (np.sqrt(v_e / (1 - 0.999 ** t)) + 1e-8)
    return E, W1, b1, W2, b2

def accuracy(X, y, E, W1, b1, W2, b2):
    emb = np.concatenate([E[j][X[:, j]] for j in range(X.shape[1])], axis=1)
    h = np.maximum(emb @ W1 + b1, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    # tuning split: last 20% in file order
    n_train = int(0.8 * n)
    X_tr, X_tune = X[:n_train], X[n_train:]
    y_tr, y_tune = y[:n_train], y[n_train:]

    best_acc, best_cfg, best_weights = -1, None, None
    for D in D_CHOICES:
        for H in H_CHOICES:
            rng = np.random.default_rng(SEED + D * 1000 + H)
            E, W1, b1, W2, b2 = train_model(X_tr, y_tr, D, H, GRID_EPOCHS, rng)
            acc = accuracy(X_tune, y_tune, E, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc, best_cfg = acc, (D, H)
                best_weights = (E, W1, b1, W2, b2)
            print(f"D={D} H={H} tune_acc={acc:.4f}", file=sys.stderr)

    D, H = best_cfg
    print(f"Best: D={D} H={H} tune_acc={best_acc:.4f}", file=sys.stderr)
    # final training on all data
    rng = np.random.default_rng(SEED)
    E, W1, b1, W2, b2 = train_model(X, y, D, H, FINAL_EPOCHS, rng)

    # precompute T[j][v] = E[j][v] @ W1[j*D:(j+1)*D]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        W1_j = W1[j * D:(j + 1) * D]
        T[j] = E[j] @ W1_j  # (256, H)

    # write weights.bin
    magic = 0x4D4C5031  # 'MLP1'
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x4D4C5031, int32 d, H, C; float T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *out = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = b1
        memcpy(h, b1, H * sizeof(float));
        // add T[j][xi[j]]
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) h[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        // out = b2 + h @ W2
        memcpy(out, b2, C * sizeof(float));
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) out[c] += hu * w[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (out[c] > out[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```