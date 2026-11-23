```hypothesis
Propose a one-hidden-layer MLP with position-specific embeddings (family "label-embmlp").
The 16-byte context benefits from non-linear interactions between positions; a small hidden layer (32–128 units) can capture such patterns while keeping inference fast. Hyperparameters (hidden size) are chosen on the last 20% of train.in. If the best validation accuracy does not exceed label-linear's 0.3375, the extra cost is not justified.
```

```python
"""label-embmlp: one-hidden-layer MLP with position-specific embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains on all train.in after selecting hidden size on the last 20% (tuning split).
Weights format: magic 0x4D4C5031, int32 d, H, C; float32 emb[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, MAX_EPOCHS = 3e-3, 1e-4, 64, 50
HIDDEN_CANDIDATES = [32, 64, 128]  # tuned on validation split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, H, rng, epochs=MAX_EPOCHS):
    """Train on (X, y) and return parameters (emb, b1, W2, b2)."""
    n, d = X.shape
    C = 65
    # Initialize parameters
    emb = rng.uniform(-0.01, 0.01, (d, 256, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            hidden = np.tile(b1, (bs, 1))  # (bs, H)
            for j in range(d):
                hidden += emb[j, xb[:, j]]  # (bs, H)
            hidden_relu = np.maximum(hidden, 0)
            scores = hidden_relu @ W2 + b2  # (bs, C)

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_scores = probs / bs  # (bs, C)

            # Backprop
            grad_W2 = hidden_relu.T @ grad_scores + L2 * W2
            grad_b2 = grad_scores.sum(axis=0)
            grad_hidden = grad_scores @ W2.T * (hidden > 0)  # (bs, H)
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, scatter-add grad_hidden to grad_emb[j, xb[:, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_hidden)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    d = X.shape[1]
    emb, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb, yb = X[s:s+BATCH], y[s:s+BATCH]
        bs = len(yb)
        hidden = np.tile(b1, (bs, 1))
        for j in range(d):
            hidden += emb[j, xb[:, j]]
        hidden = np.maximum(hidden, 0)
        scores = hidden @ W2 + b2
        pred = scores.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # Split: last 20% for validation (tuning)
    n = X.shape[0]
    val_size = n // 5
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_H, best_acc = HIDDEN_CANDIDATES[0], -1.0
    for H in HIDDEN_CANDIDATES:
        # Use a fresh RNG state for each candidate to keep deterministic
        cand_rng = np.random.default_rng(SEED + H)
        params = train_model(X_train, y_train, H, cand_rng)
        acc = evaluate(X_val, y_val, params)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on full data with best_H
    rng_full = np.random.default_rng(SEED + best_H + 1000)
    final_params = train_model(X, y, best_H, rng_full)

    emb, b1, W2, b2 = final_params
    # Write weights.bin
    magic = 0x4D4C5031  # 'MLP1'
    out = struct.pack("<4i", magic, d, best_H, C)
    out += emb.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: int32 magic(0x4D4C5031), d, H, C; float32 emb[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0 || H > 1024) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *b1 = emb + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1 + sum_j emb[j][xi[j]]
        memcpy(hidden, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += e[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // scores = hidden @ W2 + b2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```