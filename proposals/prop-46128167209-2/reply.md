```hypothesis
Family: label-embedding-mlp (position-specific byte embeddings + one ReLU hidden layer).
Why: The 16-byte context has strong positional interactions (n-grams) that a linear model on one-hot bytes cannot capture. Low-dimensional embeddings (E=8..16) drastically reduce parameters vs. one-hot (4096) while allowing a hidden layer (H=16..32) to model interactions. Inference cost is dominated by 16*E*H + H*65 multiply-adds, which can be lower than label-linear's 4096*65. Expected to beat label-linear's 0.3375 dev accuracy.
Failure: If dev accuracy ≤ 0.3375 or Ir/item ≥ 7359 (label-linear's cost), the hypothesis is not supported.
```

```python
"""label-embedding-mlp: position-specific byte embeddings followed by a single ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameters chosen on the last 20% tuning split, then trained on all data.
Weights format: magic 0x454D424C, d, E, H, C; then float32 arrays:
  emb[d*256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C]  (if H>0)
  emb[d*256*E], W_out[(d*E)*C], b_out[C]             (if H=0)"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Configs to try on tuning split: (E, H)
CONFIGS = [(8, 16), (16, 32), (8, 0), (16, 0)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(emb, W1, b1, W2, b2, X, H):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    E = emb.shape[2]
    # Embedding lookup: (n, d, E)
    embs = emb[np.arange(d), X]  # (n, d, E)
    flat = embs.reshape(n, d * E)  # (n, d*E)
    if H:
        hidden = np.maximum(flat @ W1 + b1, 0)  # (n, H)
        logits = hidden @ W2 + b2
    else:
        logits = flat @ W1 + b1  # W1 is W_out, b1 is b_out
    return logits


def train_model(X_tr, y_tr, X_val, y_val, E, H, C, rng, epochs, batch, lr, l2):
    n, d = X_tr.shape
    # Initialize parameters
    emb = rng.uniform(-0.1, 0.1, (d, 256, E)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W_out = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            embs = params[0][np.arange(d), xb]  # (batch, d, E)
            flat = embs.reshape(len(yb), d * E)
            if H:
                a = flat @ params[1] + params[2]
                h = np.maximum(a, 0)
                logits = h @ params[3] + params[4]
            else:
                logits = flat @ params[1] + params[2]
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # Backward
            if H:
                grad_logits = probs.astype(np.float32)
                grad_W2 = h.T @ grad_logits + l2 * params[3]
                grad_b2 = grad_logits.sum(0)
                grad_h = grad_logits @ params[3].T
                grad_a = grad_h * (a > 0)
                grad_W1 = flat.T @ grad_a + l2 * params[1]
                grad_b1 = grad_a.sum(0)
                grad_flat = grad_a @ params[1].T
                # Embedding gradients
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], grad_flat[:, j * E:(j + 1) * E])
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_logits = probs.astype(np.float32)
                grad_W_out = flat.T @ grad_logits + l2 * params[1]
                grad_b_out = grad_logits.sum(0)
                grad_flat = grad_logits @ params[1].T
                grad_emb = np.zeros_like(params[0])
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], grad_flat[:, j * E:(j + 1) * E])
                grads = [grad_emb, grad_W_out, grad_b_out]
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
    val_logits = forward(params[0], params[1], params[2], params[3] if H else None, params[4] if H else None, X_val, H)
    val_acc = accuracy(val_logits, y_val)
    return params, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    tr_idx = np.arange(n - tune_size)
    val_idx = np.arange(n - tune_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_params = None
    best_acc = -1.0
    best_cfg = None
    # Tuning: fewer epochs to save time
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for E, H in CONFIGS:
        cfg_rng = np.random.default_rng(SEED + E * 100 + H)
        params, acc = train_model(X_tr, y_tr, X_val, y_val, E, H, C, cfg_rng,
                                  tune_epochs, BATCH, LR, L2)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = (E, H)
    E, H = best_cfg
    # Retrain on all data with best config
    final_rng = np.random.default_rng(SEED + E * 100 + H + 1)
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params, _ = train_model(X, y, X_val, y_val, E, H, C, final_rng,
                                  final_epochs, BATCH, LR, L2)

    # Write weights.bin
    magic = 0x454D424C  # 'EMBL' in little-endian
    emb = final_params[0]
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out = struct.pack("<5i", magic, d, E, H, C)
        out += emb.astype("<f4").tobytes()
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W_out, b_out = final_params[1], final_params[2]
        out = struct.pack("<5i", magic, d, E, 0, C)
        out += emb.astype("<f4").tobytes()
        out += W_out.astype("<f4").tobytes()
        out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20;
    expected += 4L * d * 256 * E;           // emb
    if (H) {
        expected += 4L * (d * E) * H;       // W1
        expected += 4L * H;                 // b1
        expected += 4L * (long)H * C;       // W2
        expected += 4L * C;                 // b2
    } else {
        expected += 4L * (d * E) * C;       // W_out
        expected += 4L * C;                 // b_out
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    const float *W_out = W1;  // same memory when H=0
    const float *b_out = b1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        if (H) {
            // hidden = ReLU(flat * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = flat * W_out + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int f = 0; f < d * E; f++) sum += flat[f] * W_out[(long)f * C + c];
                logits[c] = sum;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```