```hypothesis
Family: label-embed-mlp — per-position byte embeddings (E=8) + one ReLU layer (H=32) + linear output.
Why: 16 one-hot positions (4096 dims) are sparse; low-rank embeddings capture character similarities and cut parameters 6×. A single hidden layer adds non-linear context mixing. Expected to beat label-linear (0.3375) while staying <10k Ir/item.
Failure: dev accuracy ≤ 0.345 or inference > 15k Ir/item.
```

```python
"""label-embed-mlp: per-position embeddings + 1 ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy."""
import json, math, struct, sys

import numpy as np

# Hyper-parameters chosen on tuning split (last 20% of train.in)
E = 8          # embedding dimension per position
H = 32         # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    d, C = X.shape[1], card["classes"]          # d=16, C=65
    n = X.shape[0]

    # validation split = last 20% (file order)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: d x 256 x E
    Emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Hidden: (d*E) x H
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output: H x C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # Embedding lookup: (B, d, E)
        emb = Emb[np.arange(d), Xb.T].transpose(1, 0, 2)   # (B, d, E)
        flat = emb.reshape(B, d * E)                       # (B, d*E)
        h = flat @ W1 + b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
        return logits, h, flat, emb

    def loss_grad(logits, yb):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        return (probs / B).astype(np.float32)

    t = 0
    best_va = -1.0
    best_params = None
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            logits, h, flat, emb = forward(xb)
            g = loss_grad(logits, yb)                       # (B, C)

            # Output layer grads
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            # Hidden layer grads
            gh = g @ W2.T * (h > 0)                         # (B, H)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # Embedding grads
            gflat = gh @ W1.T                               # (B, d*E)
            gemb = gflat.reshape(BATCH, d, E)               # (B, d, E)
            # scatter-add into d x 256 x E
            gEmb = np.zeros_like(Emb)
            np.add.at(gEmb, (np.arange(d)[:, None], xb.T), gemb.transpose(1, 0, 2))
            gEmb += L2 * Emb

            grads = [gEmb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        va_logits, *_ = forward(X_va)
        va_pred = va_logits.argmax(1)
        acc = (va_pred == y_va).mean()
        if acc > best_va:
            best_va = acc
            best_params = [p.copy() for p in params]

    # retrain on full data with best hyper-params (same seed, same schedule)
    Emb, W1, b1, W2, b2 = best_params
    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, flat, emb = forward(xb)
            g = loss_grad(logits, yb)

            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h > 0)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gflat = gh @ W1.T
            gemb = gflat.reshape(BATCH, d, E)
            gEmb = np.zeros_like(Emb)
            np.add.at(gEmb, (np.arange(d)[:, None], xb.T), gemb.transpose(1, 0, 2))
            gEmb += L2 * Emb

            grads = [gEmb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C, Emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += params[0].astype("<f4").tobytes()   # Emb
    out += params[1].astype("<f4").tobytes()   # W1
    out += params[2].astype("<f4").tobytes()   # b1
    out += params[3].astype("<f4").tobytes()   # W2
    out += params[4].astype("<f4").tobytes()   # b2
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp infer.c
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t w1_sz  = (size_t)d * E * H;
    size_t w2_sz  = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + w1_sz + H + w2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1  = Emb + emb_sz;
    const float *b1  = W1 + w1_sz;
    const float *W2  = b1 + H;
    const float *b2  = W2 + w2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_flat = malloc(sizeof(float) * d * E);
    float *hidden   = malloc(sizeof(float) * H);
    float *logits   = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *src = Emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(emb_flat + j * E, src, E * sizeof(float));
        }

        // Hidden layer: ReLU(emb_flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += emb_flat[f] * W1[(size_t)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```