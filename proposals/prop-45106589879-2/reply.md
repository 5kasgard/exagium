```hypothesis
Family: label-embmlp (embedding + one hidden layer MLP).
Why: 16 categorical byte positions benefit from learned embeddings that capture byte similarity; a hidden ReLU layer adds non-linearity missing in label-linear. This reduces parameters (embeddings + small MLP) vs full one-hot linear, so it should beat label-linear accuracy (0.3375) with comparable or lower CPU cost.
Failure: If dev accuracy ≤ 0.3375 or instructions/item exceed label-linear by >20%.
```
```python
"""label-embmlp: Embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Tunes embedding dim E, hidden dim H, lr on last 20% of train.in, then trains on all.
Weights format: magic 0x454D424D, d, E, H, C (int32); then float32 arrays:
  emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (if H>0)
  or emb[d][256][E], W2[E][C], b2[C] (if H=0)."""
import json, math, struct, sys

import numpy as np

SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Tuning grid
E_CHOICES = [16, 32, 64]
H_CHOICES = [0, 32, 64, 128]  # 0 means linear on embeddings
LR_CHOICES = [1e-3, 3e-3]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_val, y_val, d, C, E, H, lr, rng):
    n_tr = X_tr.shape[0]
    # Initialize parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    if H > 0:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W2 = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, params):
        # X: (n, d) uint8
        n = X.shape[0]
        # Sum embeddings: (n, E)
        h = np.zeros((n, E), np.float32)
        for j in range(d):
            h += params[0][j, X[:, j]]
        if H > 0:
            h = np.maximum(h @ params[1] + params[2], 0)  # ReLU
            logits = h @ params[3] + params[4]
        else:
            logits = h @ params[1] + params[2]
        return logits

    def loss_grad(logits, y):
        # logits (n, C), y (n,)
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        return probs.astype(np.float32) / n

    best_val_acc = -1.0
    best_params = None
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            n = Xb.shape[0]
            h = np.zeros((n, E), np.float32)
            for j in range(d):
                h += params[0][j, Xb[:, j]]
            if H > 0:
                a = h @ params[1] + params[2]
                h_relu = np.maximum(a, 0)
                logits = h_relu @ params[3] + params[4]
                # Backward
                g = loss_grad(logits, yb)  # (n, C)
                grad_W2 = h_relu.T @ g + L2 * params[3]
                grad_b2 = g.sum(0)
                gh = g @ params[3].T * (a > 0)  # (n, H)
                grad_W1 = h.T @ gh + L2 * params[1]
                grad_b1 = gh.sum(0)
                # Embedding gradients
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    # gh shape (n, H), but we need grad w.r.t h (n, E)
                    # dh = gh @ params[1].T  (n, E)
                    dh = gh @ params[1].T
                    # accumulate for each example
                    np.add.at(grad_emb[j], Xb[:, j], dh)
                grad_emb += L2 * params[0]
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                logits = h @ params[1] + params[2]
                g = loss_grad(logits, yb)
                grad_W2 = h.T @ g + L2 * params[1]
                grad_b2 = g.sum(0)
                # Embedding gradients
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    np.add.at(grad_emb[j], Xb[:, j], g)
                grad_emb += L2 * params[0]
                grads = [grad_emb, grad_W2, grad_b2]
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
        logits_val = forward(X_val, params)
        pred = logits_val.argmax(axis=1)
        val_acc = (pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
    return best_val_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for E in E_CHOICES:
        for H in H_CHOICES:
            for lr in LR_CHOICES:
                # Skip invalid combos
                if H == 0 and E > C:  # linear on embeddings, E can be anything but we keep reasonable
                    continue
                val_acc, params = train_eval(X_tr, y_tr, X_val, y_val, d, C, E, H, lr, rng)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cfg = (E, H, lr)
                    best_params = params
                    print(f"New best: E={E}, H={H}, lr={lr}, val_acc={val_acc:.4f}", file=sys.stderr)

    print(f"Best config: E={best_cfg[0]}, H={best_cfg[1]}, lr={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    E, H, lr = best_cfg
    # Re-initialize with same seed for reproducibility
    rng = np.random.default_rng(SEED)
    # We'll just continue training from best_params on full data for a few more epochs? Simpler: retrain from scratch on full data.
    # But we must be deterministic. We'll train on full data with same procedure but fixed epochs.
    # Use the same training function but on full data (no validation).
    n_full = X.shape[0]
    # Initialize with best_params as starting point? To be deterministic, we should train from scratch with same seed.
    # However, we already used rng for tuning. To keep deterministic, we'll create a new rng with same seed and train on full data.
    # But the tuning loop consumed random numbers. So we need to reset rng to a known state.
    # Simpler: we'll just do a final training on full data with the same hyperparameters, using a fresh rng seeded with SEED.
    # This is deterministic because the whole script is deterministic.
    rng_full = np.random.default_rng(SEED)
    # We'll run a fixed number of epochs on full data.
    # We'll reuse train_eval but without validation, just train for fixed epochs.
    # Let's write a simple training loop for full data.
    emb = (rng_full.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    if H > 0:
        W1 = (rng_full.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng_full.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W2 = (rng_full.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def loss_grad(logits, y):
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        return probs.astype(np.float32) / n

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))):
        order = rng_full.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            n = Xb.shape[0]
            h = np.zeros((n, E), np.float32)
            for j in range(d):
                h += params[0][j, Xb[:, j]]
            if H > 0:
                a = h @ params[1] + params[2]
                h_relu = np.maximum(a, 0)
                logits = h_relu @ params[3] + params[4]
                g = loss_grad(logits, yb)
                grad_W2 = h_relu.T @ g + L2 * params[3]
                grad_b2 = g.sum(0)
                gh = g @ params[3].T * (a > 0)
                grad_W1 = h.T @ gh + L2 * params[1]
                grad_b1 = gh.sum(0)
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    dh = gh @ params[1].T
                    np.add.at(grad_emb[j], Xb[:, j], dh)
                grad_emb += L2 * params[0]
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                logits = h @ params[1] + params[2]
                g = loss_grad(logits, yb)
                grad_W2 = h.T @ g + L2 * params[1]
                grad_b2 = g.sum(0)
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    np.add.at(grad_emb[j], Xb[:, j], g)
                grad_emb += L2 * params[0]
                grads = [grad_emb, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    E, H = best_cfg[0], best_cfg[1]
    magic = 0x454D424D  # "MBME"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += params[0].astype("<f4").tobytes()  # emb
    if H > 0:
        out += params[1].astype("<f4").tobytes()  # W1
        out += params[2].astype("<f4").tobytes()  # b1
        out += params[3].astype("<f4").tobytes()  # W2
        out += params[4].astype("<f4").tobytes()  # b2
    else:
        out += params[1].astype("<f4").tobytes()  # W2
        out += params[2].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x454D424D, d, E, H, C (int32 little-endian).
 * Then float32 arrays: emb[d][256][E], (if H>0: W1[E][H], b1[H], W2[H][C], b2[C] else: W2[E][C], b2[C]). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size
    long expected = 20;
    expected += (long)d * 256 * E;           // emb
    if (H > 0) {
        expected += (long)E * H;             // W1
        expected += H;                       // b1
        expected += (long)H * C;             // W2
        expected += C;                       // b2
    } else {
        expected += (long)E * C;             // W2
        expected += C;                       // b2
    }
    expected *= 4; // float32 bytes
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (H > 0 ? (long)E * H : 0);
    const float *W2 = b1 + (H > 0 ? H : 0);
    const float *b2 = W2 + (H > 0 ? (long)H * C : (long)E * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *hidden = H > 0 ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e];
        }

        if (H > 0) {
            // Hidden layer: ReLU(h @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int e = 0; e < E; e++) sum += h[e] * W1[(long)e * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                scores[c] = sum;
            }
        } else {
            // Linear on embeddings
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int e = 0; e < E; e++) sum += h[e] * W2[(long)e * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(hidden); free(scores);
    return 0;
}
```