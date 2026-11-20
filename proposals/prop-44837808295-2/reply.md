```hypothesis
Family: label-emb (per-position byte embeddings + linear classifier).  
Why: One-hot linear (label-linear) uses 4096 sparse features; embeddings compress each byte into a dense vector (E=8..32), capturing character similarities and reducing parameters. The concatenated 16*E features feed a linear layer, preserving position. This should exceed label-linear accuracy (0.3375) while being faster (smaller matmul).  
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear (7359.9), the hypothesis is invalid.
```

```python
"""label-emb: per-position byte embeddings + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all.
Weights format: magic(0x454D424C), d, E, C; float32 emb[d*256*E], W[(d*E)*C], b[C]."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
E_CHOICES = [4, 8, 16, 32]
LR_CHOICES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # max samples per epoch for early stopping


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X_tr, y_tr, X_val, y_val, d, C, E, lr, rng):
    """Train embeddings + linear, return (emb, W, b, val_acc)."""
    n_tr = X_tr.shape[0]
    # emb shape (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W = (rng.normal(0, 0.01, (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_val_acc = -1.0
    best_params = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bsz = xb.shape[0]
            # Forward: gather embeddings
            # emb_lookup shape (bsz, d, E)
            emb_lookup = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (bsz, d, E)
            flat = emb_lookup.reshape(bsz, d * E)  # (bsz, d*E)
            logits = flat @ W + b  # (bsz, C)
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz  # (bsz, C)
            # Gradients
            grad_W = flat.T @ probs + L2 * W  # (d*E, C)
            grad_b = probs.sum(axis=0)        # (C,)
            grad_flat = probs @ W.T           # (bsz, d*E)
            grad_emb_lookup = grad_flat.reshape(bsz, d, E)  # (bsz, d, E)
            # Accumulate embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # xb[:, j] are byte indices for position j
                np.add.at(grad_emb[j], xb[:, j], grad_emb_lookup[:, j, :])
            grad_emb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [(emb, grad_emb, m_emb, v_emb),
                                      (W, grad_W, m_W, v_W),
                                      (b, grad_b, m_b, v_b)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # Validation accuracy
        emb_val = emb[np.arange(d), X_val.T].transpose(1, 0, 2)
        flat_val = emb_val.reshape(X_val.shape[0], d * E)
        logits_val = flat_val @ W + b
        pred = logits_val.argmax(axis=1)
        val_acc = (pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (emb.copy(), W.copy(), b.copy())
    return best_params + (best_val_acc,)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    n = X.shape[0]
    val_size = n // 5
    X_tr, X_val = X[:-val_size], X[-val_size:]
    y_tr, y_val = y[:-val_size], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_overall = None
    best_acc = -1.0
    best_E = best_lr = None

    for E in E_CHOICES:
        for lr in LR_CHOICES:
            rng_config = np.random.default_rng(SEED + E * 1000 + int(lr * 10000))
            emb, W, b, val_acc = train_model(X_tr, y_tr, X_val, y_val, d, C, E, lr, rng_config)
            if val_acc > best_acc:
                best_acc = val_acc
                best_overall = (emb, W, b)
                best_E, best_lr = E, lr
    print(f"Chosen E={best_E}, lr={best_lr}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    rng_full = np.random.default_rng(SEED + best_E * 1000 + int(best_lr * 10000))
    emb, W, b, _ = train_model(X, y, X_val[:1], y_val[:1], d, C, best_E, best_lr, rng_full)
    # (We pass a dummy validation set just to reuse the function; the returned params are from full training)

    # Write weights
    magic = 0x454D424C  # "LBME" little-endian
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x454D424C), d, E, C; float32 emb[d*256*E], W[(d*E)*C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E < 1 || E > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding vector
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < d * E; f++) sum += concat[f] * W[(long)f * C + c];
            scores[c] = sum;
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(concat); free(scores);
    return 0;
}
```