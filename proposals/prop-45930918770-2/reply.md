```hypothesis
Family: label-embed (factorized linear model with shared byte embeddings and position-specific output projections).
Why: The 16-byte context has strong positional patterns and byte similarities (e.g., letters cluster). A shared 16-dim embedding per byte value (256×16) plus 16 position-specific linear maps (16×16×65) captures this with ~20k parameters (vs 266k for label-linear), reducing overfitting and speeding up inference. Embedding similarity transfers knowledge across positions and byte values.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 7359 Ir/item.
```

```python
"""label-embed: factorized linear model with shared byte embeddings and position-specific classifiers.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Chooses embedding dimension D on tuning split (last 20% of train.in), then trains on all train.
Exports: magic 0x454D4245, d=16, D, C; float32 emb[256][D], W[16][D][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Candidate embedding dimensions to try on tuning split
D_CANDIDATES = [8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def train_eval(X_tr, y_tr, X_val, y_val, D, C, rng):
    """Train on (X_tr, y_tr), return (weights_dict, val_acc)."""
    n_tr, d = X_tr.shape
    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d, D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute one-hot indices for speed? We'll do embedding lookup per batch.
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb = X_tr[idx]          # (B, d)
            yb = y_tr[idx]          # (B,)
            B = len(yb)

            # Forward: embed each position and accumulate
            # h = sum_j emb[xb[:, j]] @ W[j] + b   shape (B, C)
            h = np.zeros((B, C), np.float32)
            for j in range(d):
                e = emb[xb[:, j]]           # (B, D)
                h += e @ W[j]               # (B, C)
            h += b

            # Softmax gradient
            h_max = h.max(axis=1, keepdims=True)
            exp_h = np.exp(h - h_max)
            probs = exp_h / exp_h.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_out = (probs / B).astype(np.float32)   # (B, C)

            # Gradients
            grad_W = np.zeros_like(W)
            grad_emb = np.zeros_like(emb)
            grad_b = grad_out.sum(axis=0)

            for j in range(d):
                e = emb[xb[:, j]]           # (B, D)
                grad_W[j] = e.T @ grad_out  # (D, C)
                # scatter-add to grad_emb
                np.add.at(grad_emb, xb[:, j], grad_out @ W[j].T)

            # L2 regularization on W and emb (not on b)
            grad_W += L2 * W
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Evaluate on validation
    n_val = X_val.shape[0]
    val_pred = np.zeros(n_val, np.int64)
    for s in range(0, n_val, BATCH):
        idx = slice(s, s + BATCH)
        xb = X_val[idx]
        h = np.zeros((len(xb), C), np.float32)
        for j in range(d):
            h += emb[xb[:, j]] @ W[j]
        h += b
        val_pred[idx] = h.argmax(axis=1)
    val_acc = accuracy(y_val, val_pred)

    return {"emb": emb, "W": W, "b": b}, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    # Tuning split is last 20% in file order (no shuffle)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    best_weights = None

    for D in D_CANDIDATES:
        weights, acc = train_eval(X_tr, y_tr, X_val, y_val, D, C, rng)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_weights = weights

    # Retrain on full data with best_D
    rng_full = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_weights, _ = train_eval(X, y, X[:1], y[:1], best_D, C, rng_full)  # dummy val

    # Pack weights: magic, d, D, C; then emb[256*D], W[d*D*C], b[C] all float32 little-endian
    emb = final_weights["emb"].astype("<f4")
    W = final_weights["W"].astype("<f4")
    b = final_weights["b"].astype("<f4")
    magic = 0x454D4245  # "EMBE"
    header = struct.pack("<4i", magic, d, best_D, C)
    out = header + emb.tobytes() + W.tobytes() + b.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D4245), d, D, C (int32 each);
 * float32 emb[256][D], W[d][D][C], b[C] (row-major). */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (256L * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + 256L * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *tmp = malloc(sizeof(float) * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // init scores with bias
        memcpy(scores, b, C * sizeof(float));

        // accumulate position contributions
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * D;
            const float *Wj = W + (long)j * D * C;
            // tmp = e (size D)  -- we can directly do e^T @ Wj
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int u = 0; u < D; u++) {
                    sum += e[u] * Wj[(long)u * C + c];
                }
                scores[c] += sum;
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(scores); free(tmp);
    return 0;
}
```