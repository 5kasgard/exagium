```hypothesis
Family: label-embed (positional embeddings + optional one-hidden-layer MLP).
Why: The 16-byte context has strong positional patterns; learned embeddings capture byte similarities (e.g., letters vs punctuation) that one-hot cannot, and a small ReLU layer adds non-linearity. With embedding dim 8 and hidden 64 the model has ~20k parameters – still tiny – so inference stays near label-linear speed while accuracy should rise.
Failure signal: dev accuracy ≤ 0.3375 or instructions/item ≥ 1.5× label-linear.
```

```python
"""label-embed: positional embeddings + optional 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D4244 ('EMBD'), d, E, H, C
  float32 emb[d][256][E]
  float32 W1[d*E][H or C], b1[H or C]
  if H>0: float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

# Grid searched on tuning split (last 20% of train.in)
GRID = [
    {"E": 4, "H": 0},
    {"E": 8, "H": 0},
    {"E": 4, "H": 32},
    {"E": 8, "H": 32},
    {"E": 8, "H": 64},
]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
TUNE_EPOCHS = 10          # epochs for grid search on 80% train
FULL_EPOCHS_MULT = 1.0    # multiplier for final training epochs


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, rng):
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    in_dim = d * E
    out_dim = H if H else C
    W1 = rng.uniform(-1, 1, (in_dim, out_dim)).astype(np.float32) * math.sqrt(6.0 / in_dim)
    b1 = np.zeros(out_dim, np.float32)
    if H:
        W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
        b2 = np.zeros(C, np.float32)
        return [emb, W1, b1, W2, b2]
    return [emb, W1, b1]


def forward(params, X, d, E, H, C):
    n = X.shape[0]
    emb = params[0]
    # embed: (n, d, E)
    x_emb = emb[X.reshape(-1), np.arange(256)].reshape(n, d, E)  # fancy indexing trick
    x_flat = x_emb.reshape(n, d * E)
    W1, b1 = params[1], params[2]
    h = x_flat @ W1 + b1
    if H:
        h = np.maximum(h, 0)
        W2, b2 = params[3], params[4]
        logits = h @ W2 + b2
    else:
        logits = h
    return logits, x_flat, h if H else None


def loss_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # dL/dlogits


def backward(params, X, y, d, E, H, C, x_flat, h):
    n = X.shape[0]
    g_logits = loss_grad(*forward(params, X, d, E, H, C)[:1], y)  # recompute logits for grad
    emb, W1, b1 = params[0], params[1], params[2]
    if H:
        W2, b2 = params[3], params[4]
        g_W2 = h.T @ g_logits + L2 * W2
        g_b2 = g_logits.sum(0)
        g_h = g_logits @ W2.T
        g_h *= (h > 0)
        g_W1 = x_flat.T @ g_h + L2 * W1
        g_b1 = g_h.sum(0)
        # embedding gradients
        g_emb = np.zeros_like(emb)
        g_x_flat = g_h @ W1.T
        g_x_emb = g_x_flat.reshape(n, d, E)
        np.add.at(g_emb, X, g_x_emb)
        return [g_emb, g_W1, g_b1, g_W2, g_b2]
    else:
        g_W1 = x_flat.T @ g_logits + L2 * W1
        g_b1 = g_logits.sum(0)
        g_emb = np.zeros_like(emb)
        g_x_flat = g_logits @ W1.T
        g_x_emb = g_x_flat.reshape(n, d, E)
        np.add.at(g_emb, X, g_x_emb)
        return [g_emb, g_W1, g_b1]


def accuracy(params, X, y, d, E, H, C):
    logits = forward(params, X, d, E, H, C)[0]
    return (logits.argmax(1) == y).mean()


def train_loop(params, X, y, d, E, H, C, epochs, rng):
    n = X.shape[0]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            grads = backward(params, xb, yb, d, E, H, C, *forward(params, xb, d, E, H, C)[1:])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def pack_weights(params, d, E, H, C):
    magic = 0x454D4244
    emb, W1, b1 = params[0], params[1], params[2]
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2, b2 = params[3], params[4]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = len(X)
    tune_size = n // 5
    tr_X, tr_y = X[:-tune_size], y[:-tune_size]
    tu_X, tu_y = X[-tune_size:], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None

    for cfg in GRID:
        E, H = cfg["E"], cfg["H"]
        cfg_rng = np.random.default_rng(SEED + hash((E, H)) % 2**32)
        params = init_params(d, E, H, C, cfg_rng)
        epochs = min(MAX_EPOCHS, max(1, math.ceil(TUNE_EPOCHS * SAMPLES / len(tr_X))))
        params = train_loop(params, tr_X, tr_y, d, E, H, C, epochs, cfg_rng)
        acc = accuracy(params, tu_X, tu_y, d, E, H, C)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, params

    # Retrain best config on full data
    E, H = best_cfg["E"], best_cfg["H"]
    full_rng = np.random.default_rng(SEED + 12345)
    params = init_params(d, E, H, C, full_rng)
    epochs = min(MAX_EPOCHS, max(1, math.ceil(FULL_EPOCHS_MULT * SAMPLES / n)))
    params = train_loop(params, X, y, d, E, H, C, epochs, full_rng)

    open(sys.argv[4], "wb").write(pack_weights(params, d, E, H, C))


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E < 1 || E > 32 || H < 0 || H > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    size_t emb_sz = (size_t)d * 256 * E;
    size_t w1_sz = (size_t)d * E * (H ? H : C);
    size_t b1_sz = H ? H : C;
    size_t w2_sz = H ? (size_t)H * C : 0;
    size_t b2_sz = H ? C : 0;
    long expected = 20 + 4L * (emb_sz + w1_sz + b1_sz + w2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + w1_sz;
    const float *W2 = H ? b1 + b1_sz : NULL;
    const float *b2 = H ? W2 + w2_sz : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *acc = malloc(sizeof(float) * (H ? H : C));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        if (H) {
            for (int h = 0; h < H; h++) acc[h] = b1[h];
        } else {
            for (int c = 0; c < C; c++) acc[c] = b1[c];
        }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            if (H) {
                for (int h = 0; h < H; h++) {
                    float sum = 0.0f;
                    for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((size_t)j * E + ee) * H + h];
                    acc[h] += sum;
                }
            } else {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((size_t)j * E + ee) * C + c];
                    acc[c] += sum;
                }
            }
        }

        if (H) {
            for (int h = 0; h < H; h++) hidden[h] = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) logits[c] = acc[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```