```hypothesis
Family: label-mlp (positional embeddings + one hidden ReLU layer).
Why: The linear model cannot capture interactions between context positions. A small MLP on concatenated positional embeddings can model n-gram-like features while keeping inference fast via table lookups and dense matrix multiplies. The embedding dimension and hidden size are tuned on the held-out 20% of train data.
Failure: If accuracy on dev does not exceed label-linear's 0.3375, or if inference cost exceeds label-linear by more than 2x.
```

```python
"""label-mlp: positional embeddings + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single thread. Tunes E,H on last 20% of train, then trains on all.
Exports: magic 0x4D4C5031, d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 12345
LR_GRID = [1e-3, 3e-3]
E_GRID = [16, 32]
H_GRID = [64, 128]
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
L2 = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def train_model(X_tr, y_tr, X_va, y_va, d, C, E, H, lr, rng):
    # Initialize parameters
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_tr = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    best_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            # emb lookup: (batch, d, E)
            emb_out = emb[:, xb.T, :].transpose(1, 0, 2)  # (batch, d, E)
            flat = emb_out.reshape(len(yb), -1)           # (batch, d*E)
            h_pre = flat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T
            grad_h[h_pre <= 0] = 0
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            # Gradient wrt embeddings: (batch, d, E)
            grad_flat = grad_h @ W1.T
            grad_emb_out = grad_flat.reshape(len(yb), d, E)
            grad_emb = np.zeros_like(emb)
            # scatter add: for each position j, for each example i, add grad_emb_out[i,j] to grad_emb[j, xb[i,j]]
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_out[:, j])
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        emb_out_va = emb[:, X_va.T, :].transpose(1, 0, 2)
        flat_va = emb_out_va.reshape(len(y_va), -1)
        h_va = np.maximum(flat_va @ W1 + b1, 0)
        logits_va = h_va @ W2 + b2
        acc = accuracy(logits_va, y_va)
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
    return best_params, best_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_overall = None
    best_score = -1.0
    best_hp = None

    for E in E_GRID:
        for H in H_GRID:
            for lr in LR_GRID:
                params, score = train_model(X_tr, y_tr, X_va, y_va, d, C, E, H, lr, rng)
                if score > best_score:
                    best_score = score
                    best_overall = params
                    best_hp = (E, H, lr)

    # Retrain on full data with best hyperparameters
    E, H, lr = best_hp
    rng = np.random.default_rng(SEED)  # reset for reproducible final training
    final_params, _ = train_model(X, y, X, y, d, C, E, H, lr, rng)
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x4D4C5031  # 'MPL1'
    header = struct.pack("<4i", magic, d, E, H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x4D4C5031, d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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