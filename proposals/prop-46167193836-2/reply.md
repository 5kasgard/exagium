```hypothesis
Family: label-emb (positional embeddings + linear/ReLU classifier). Each of the 16 context bytes is mapped to a D-dimensional vector via a learned lookup table; the 16 vectors are summed and fed to a linear layer (or one ReLU hidden layer). This reduces parameters from 266k (label-linear one-hot) to ~33k (D=8) while capturing byte similarities, improving generalization on the small tinyshakespeare train set. Inference cost drops because we sum 16×D floats instead of adding 16×C one-hot columns. We expect dev accuracy >0.3375 and lower Ir/item than label-linear. If dev accuracy ≤0.3375 or Ir/item ≥7359, the hypothesis fails.
```

```python
"""label-emb: positional embedding model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of 16 positions with a 256×D embedding table, sums them, then linear (or +ReLU hidden).
Tunes D∈{4,8,16} and H∈{0,16,32} on last 20% of train.in, then retrains on all train."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_CHOICES = [4, 8, 16]
H_CHOICES = [0, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t. logits."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_model(X, y, d, C, D, H, rng, n_tune=None):
    """Train embeddings + optional hidden layer. Returns (emb, W1, b1, W2, b2)."""
    n = X.shape[0]
    # embeddings: (d, 256, D)
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward: sum embeddings for each position
            summed = np.zeros((len(xb), D), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j]]
            if H:
                h = summed @ W1 + b1
                h_relu = np.maximum(h, 0)
                logits = h_relu @ W2 + b2
            else:
                logits = summed @ W1 + b1
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # backward
            if H:
                g_W2 = h_relu.T @ g
                g_b2 = g.sum(0)
                g_h = g @ W2.T
                g_h[h <= 0] = 0.0
                g_W1 = summed.T @ g_h
                g_b1 = g_h.sum(0)
                g_summed = g_h @ W1.T
            else:
                g_W1 = summed.T @ g
                g_b1 = g.sum(0)
                g_summed = g @ W1.T
            # embedding gradients: scatter-add
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_summed)
            # L2 regularization on all params except embeddings? Apply to all.
            grads = [g_emb + L2 * emb]
            if H:
                grads += [g_W1 + L2 * W1, g_b1, g_W2 + L2 * W2, g_b2]
            else:
                grads += [g_W1 + L2 * W1, g_b1]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, d, C, D, H, params):
    emb = params[0]
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W1, b1 = params[1], params[2]
    summed = np.zeros((X.shape[0], D), np.float32)
    for j in range(d):
        summed += emb[j, X[:, j]]
    if H:
        h = np.maximum(summed @ W1 + b1, 0)
        logits = h @ W2 + b2
    else:
        logits = summed @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_D, best_H = -1, None, None, None
    for D in D_CHOICES:
        for H in H_CHOICES:
            params = train_model(X_train, y_train, d, C, D, H, rng)
            acc = evaluate(X_tune, y_tune, d, C, D, H, params)
            if acc > best_acc:
                best_acc, best_params, best_D, best_H = acc, params, D, H
                print(f"tune: D={D} H={H} acc={acc:.4f}", file=sys.stderr)

    # retrain on all data with best hyperparameters
    rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, d, C, best_D, best_H, rng)
    emb, *rest = final_params
    if best_H:
        W1, b1, W2, b2 = rest
    else:
        W1, b1 = rest
        W2, b2 = None, None

    # write weights: magic, d, D, H, C; then emb[d][256][D], W1, b1, (W2, b2)
    magic = 0x4C424D45  # 'EMBL'
    out = struct.pack("<5i", magic, d, best_D, best_H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: int32 magic(0x4C424D45), d, D, H, C;
 * float32 emb[d][256][D], W1[D][H or C], b1[H or C]; if H>0: W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)D * (H ? H : C) + (H ? H : C) + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * (H ? H : C);
    const float *W2 = H ? b1 + (H ? H : C) : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(summed, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) summed[v] += e[v];
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float val = b1[h];
                for (int v = 0; v < D; v++) val += summed[v] * W1[v * H + h];
                hidden[h] = val > 0.0f ? val : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float val = b2[c];
                for (int h = 0; h < H; h++) val += hidden[h] * W2[h * C + c];
                scores[c] = val;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float val = b1[c];
                for (int v = 0; v < D; v++) val += summed[v] * W1[v * C + c];
                scores[c] = val;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```