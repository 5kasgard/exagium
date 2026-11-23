```hypothesis
Family: label-embmlp (embedding MLP). Each of the 14 byte features gets a learned embedding vector; the embeddings are summed, passed through one ReLU hidden layer, then a 2-class output. This captures feature interactions better than linear while staying far cheaper than GBT or large MLP. The embedding tables (14×256×E) dominate parameters but are cache-friendly at inference. If dev accuracy does not exceed label-linear (0.8524) or CPU cost per item exceeds label-linear by >2×, the idea fails.
```
```python
"""label-embmlp: per-feature embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format (little-endian):
  magic 0x454D4244 ('DBME'), int32 d, E, H, C
  float32 emb[d][256][E]
  float32 W1[E][H], b1[H]
  float32 W2[H][C], b2[C]
"""
import json, math, struct, sys

import numpy as np

# Tuning grid (searched on last 20% of train.in in file order)
E_GRID = [8, 16, 32]
H_GRID = [16, 32, 64]
SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100
TUNE_EPOCHS = 20          # epochs per config during tuning
FINAL_EPOCHS_MULT = 1.5   # train longer on full data

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def tune_split(X, y):
    n = X.shape[0]
    cut = int(0.8 * n)
    return X[:cut], y[:cut], X[cut:], y[cut:]

def init_params(d, E, H, C, rng):
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(X, emb, W1, b1, W2, b2):
    n = X.shape[0]
    emb_sum = np.zeros((n, emb.shape[2]), np.float32)
    for j in range(emb.shape[0]):
        emb_sum += emb[j, X[:, j]]
    hidden = emb_sum @ W1 + b1
    np.maximum(hidden, 0, out=hidden)
    logits = hidden @ W2 + b2
    return emb_sum, hidden, logits

def loss_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    return probs.astype(np.float32)

def train_epoch(X, y, emb, W1, b1, W2, b2, state, rng, t):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb, yb = X[idx], y[idx]
        emb_sum, hidden, logits = forward(xb, emb, W1, b1, W2, b2)
        dlogits = loss_grad(logits, yb)

        # gradients
        dW2 = hidden.T @ dlogits + L2 * W2
        db2 = dlogits.sum(0)
        dhidden = dlogits @ W2.T
        dhidden[hidden == 0] = 0.0
        dW1 = emb_sum.T @ dhidden + L2 * W1
        db1 = dhidden.sum(0)
        demb_sum = dhidden @ W1.T

        demb = np.zeros_like(emb)
        for j in range(emb.shape[0]):
            np.add.at(demb[j], xb[:, j], demb_sum)

        grads = [demb, dW1, db1, dW2, db2]
        params = [emb, W1, b1, W2, b2]
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            t += 1
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t

def accuracy(X, y, emb, W1, b1, W2, b2):
    _, _, logits = forward(X, emb, W1, b1, W2, b2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]

    rng = np.random.default_rng(SEED)

    # Tuning on last 20% (by file order)
    X_tr, y_tr, X_val, y_val = tune_split(X, y)
    best_acc, best_cfg, best_params = -1, None, None

    for E in E_GRID:
        for H in H_GRID:
            emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in [emb, W1, b1, W2, b2]]
            t = 0
            for _ in range(TUNE_EPOCHS):
                t = train_epoch(X_tr, y_tr, emb, W1, b1, W2, b2, state, rng, t)
            acc = accuracy(X_val, y_val, emb, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc, best_cfg = acc, (E, H)
                best_params = [p.copy() for p in [emb, W1, b1, W2, b2]]

    E, H = best_cfg
    emb, W1, b1, W2, b2 = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in [emb, W1, b1, W2, b2]]
    t = 0
    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n * FINAL_EPOCHS_MULT))
    for _ in range(max_epochs):
        t = train_epoch(X, y, emb, W1, b1, W2, b2, state, rng, t)

    # Write weights
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class index). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E <= 0 || H <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(emb_sum, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) emb_sum[e_idx] += e[e_idx];
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e_idx = 0; e_idx < E; e_idx++) sum += emb_sum[e_idx] * W1[(long)e_idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```