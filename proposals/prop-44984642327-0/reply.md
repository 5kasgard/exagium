```hypothesis
Family: label-emb-mlp. Learns a small embedding per feature value (256 per column) summed into a dense vector,
then a one-hidden-layer ReLU MLP to 2 logits. Captures non-linear feature interactions at lower parameter count
than full one-hot MLP. Should beat label-linear (0.8524) on adult dev. Failure: dev accuracy ≤ 0.8524 or
inference cost > 2× label-linear (Ir/item > 1500).
```

```python
"""label-emb-mlp: embedding + 1-hidden-layer MLP for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Architecture: per-feature embedding (256, E) -> sum -> ReLU(H) -> linear(2).
Exported weights.bin: magic, d, E, H, C; embeddings[d][256][E]; W1[E][H]; b1[H]; W2[H][2]; b2[2] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid (tuned on tuning split)
EMB_DIMS = [4, 8, 16]
HIDDEN_DIMS = [8, 16, 32]
LRS = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_va, y_va, d, E, H, C, lr, rng):
    # embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n_tr = X_tr.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    t = 0
    for _ in range(steps):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            # sum embeddings: (B, E)
            summed = np.zeros((len(idx), E), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j]]
            h_pre = summed @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T
            grad_h[h_pre <= 0] = 0.0
            grad_W1 = summed.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            grad_summed = grad_h @ W1.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_summed)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    summed_va = np.zeros((X_va.shape[0], E), np.float32)
    for j in range(d):
        summed_va += emb[j, X_va[:, j]]
    h_va = np.maximum(summed_va @ W1 + b1, 0)
    logits_va = h_va @ W2 + b2
    preds = logits_va.argmax(axis=1)
    acc = (preds == y_va).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_cfg = None
    for E in EMB_DIMS:
        for H in HIDDEN_DIMS:
            for lr in LRS:
                rng = np.random.default_rng(SEED)
                acc, params = train_eval(X_tr, y_tr, X_va, y_va, d, E, H, C, lr, rng)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_cfg = (E, H, lr)
    # retrain on full data with best config
    E, H, lr = best_cfg
    rng = np.random.default_rng(SEED)
    _, final_params = train_eval(X, y, X[:1], y[:1], d, E, H, C, lr, rng)  # dummy val
    emb, W1, b1, W2, b2 = final_params

    # write weights.bin
    magic = 0x454D4231  # 'EMB1'
    out = struct.pack("<5i", magic, d, E, H, C)
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
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class index (argmax).
 * Weights format (little-endian):
 *   int32 magic=0x454D4231, d, E, H, C
 *   float32 emb[d][256][E]
 *   float32 W1[E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d <= 0 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // sum embeddings
        for (int e = 0; e < E; e++) summed[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) summed[e] += eptr[e];
        }
        // hidden layer
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int e = 0; e < E; e++) acc += summed[e] * W1[(long)e * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }
        // output layer
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
            logits[c] = acc;
        }
        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```