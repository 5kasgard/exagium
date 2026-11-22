```hypothesis
Family: label-embed-mlp (shared byte embeddings + positional embeddings, one ReLU hidden layer).
Why: The task is character-level language modeling; shared embeddings capture semantic similarity between bytes (e.g., letters cluster) while positional embeddings retain order information. This reduces parameters vs. per-position one-hot (256*E + 16*E vs 16*256*W), allowing a larger hidden layer (H=256) within the same budget, which should improve accuracy over label-linear (0.3375). The model is still a simple MLP, so inference remains fast (O(E*16 + E*H + H*C)).
Failure: If dev accuracy ≤ 0.3375 or instructions/item exceed label-linear by >2×, the family does not justify its complexity.
```

```python
"""label-embed-mlp: shared byte embeddings + positional embeddings, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses hyperparams on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the tuning split (last 20% of train.in)
GRID = [
    {"E": 32, "H": 128, "lr": 3e-3, "epochs": 30},
    {"E": 32, "H": 256, "lr": 3e-3, "epochs": 30},
    {"E": 64, "H": 128, "lr": 3e-3, "epochs": 30},
    {"E": 64, "H": 256, "lr": 1e-3, "epochs": 40},
]
SEED = 1
L2 = 1e-5
BATCH = 128
CLIP = 1.0

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(E, H, C, rng):
    # shared_emb: 256 x E, pos_emb: 16 x E
    shared = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    pos = (rng.normal(0, 0.01, (16, E))).astype(np.float32)
    # hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [shared, pos, W1, b1, W2, b2]

def forward(params, X, training=False):
    shared, pos, W1, b1, W2, b2 = params
    n = X.shape[0]
    # embedding lookup and sum over positions
    h = np.zeros((n, shared.shape[1]), np.float32)
    for j in range(16):
        h += shared[X[:, j]] + pos[j]
    # hidden layer
    a = h @ W1 + b1
    h2 = np.maximum(a, 0)
    # output
    logits = h2 @ W2 + b2
    if training:
        return logits, h, h2, a
    return logits

def backward(params, X, y, logits, h, h2, a):
    shared, pos, W1, b1, W2, b2 = params
    n = X.shape[0]
    # softmax gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # output layer grads
    gW2 = h2.T @ probs
    gb2 = probs.sum(0)
    # hidden layer grads
    gh2 = probs @ W2.T
    gh2[a <= 0] = 0.0
    gW1 = h.T @ gh2
    gb1 = gh2.sum(0)
    # embedding grads
    g_shared = np.zeros_like(shared)
    g_pos = np.zeros_like(pos)
    gh = gh2 @ W1.T
    for j in range(16):
        np.add.at(g_shared, X[:, j], gh)
        g_pos[j] += gh.sum(0)
    return [g_shared, g_pos, gW1, gb1, gW2, gb2]

def accuracy(params, X, y):
    logits = forward(params, X)
    return (logits.argmax(1) == y).mean()

def train_one(config, X_tr, y_tr, X_val, y_val):
    rng = np.random.default_rng(SEED)
    params = init_params(config["E"], config["H"], 65, rng)
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X_tr.shape[0]
    best_val = 0.0
    best_params = None
    for epoch in range(config["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, h, h2, a = forward(params, Xb, training=True)
            grads = backward(params, Xb, yb, logits, h, h2, a)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g += L2 * p
                # clip
                norm = np.sqrt((g*g).sum())
                if norm > CLIP:
                    g *= CLIP / norm
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= config["lr"] * m_hat / (np.sqrt(v_hat) + 1e-8)
        val_acc = accuracy(params, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    # split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_config = None
    best_val = -1.0
    best_params = None
    for cfg in GRID:
        params, val = train_one(cfg, X_tr, y_tr, X_val, y_val)
        print(f"Config {cfg}: val_acc={val:.4f}", file=sys.stderr)
        if val > best_val:
            best_val = val
            best_config = cfg
            best_params = params
    print(f"Best config: {best_config}, val_acc={best_val:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    rng = np.random.default_rng(SEED)
    params = init_params(best_config["E"], best_config["H"], C, rng)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(best_config["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h, h2, a = forward(params, Xb, training=True)
            grads = backward(params, Xb, yb, logits, h, h2, a)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g += L2 * p
                norm = np.sqrt((g*g).sum())
                if norm > CLIP:
                    g *= CLIP / norm
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= best_config["lr"] * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic, d, E, H, C, then float32 arrays in order
    E, H = best_config["E"], best_config["H"]
    magic = 0x454D4231  # "EMB1"
    header = struct.pack("<5i", magic, d, E, H, C)
    shared, pos, W1, b1, W2, b2 = best_params
    out = header
    out += shared.astype("<f4").tobytes()
    out += pos.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20
        + 4L * 256 * E
        + 4L * d * E
        + 4L * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *shared = (const float *)(hd + 5);
    const float *pos_emb = shared + 256 * E;
    const float *W1 = pos_emb + d * E;
    const float *b1 = W1 + E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *h2 = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* embedding sum */
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *s = shared + xi[j] * E;
            const float *p = pos_emb + j * E;
            for (int e = 0; e < E; e++) h[e] += s[e] + p[e];
        }
        /* hidden layer */
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e = 0; e < E; e++) sum += h[e] * W1[e * H + u];
            h2[u] = sum > 0.0f ? sum : 0.0f;
        }
        /* output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h2[u] * W2[u * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(h2); free(scores);
    if (x) free(x);
    free(hd);
    return 0;
}
```