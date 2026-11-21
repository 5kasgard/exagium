```hypothesis
Family: label-fm (Factorization Machine for categorical sequences).
Why: Next-char prediction relies heavily on pairwise interactions between recent characters (n-grams). Linear models miss these; MLPs with few hidden units (mlp16) cannot capture all O(d^2) interactions. FM explicitly models all pairwise interactions via k latent factors (O(k*d) params), is expressive for language, and has fast inference (sum embeddings, compute pairwise, linear). 
Failure signal: Dev accuracy ≤ 33.75% (label-linear) or large train-dev gap (>5%) indicating overfit.
```

```python
"""label-fm: Factorization Machine for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes k (latent factors) and lr on last 20% of train, then retrains on all.
Weights format: magic 0x464D3130, d, k, C; emb[d][256][k] (float32); W_out[2k][C]; bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
L2 = 1e-4
SAMPLES = 600000
MAX_EPOCHS = 100
# Tuning grid
K_GRID = [8, 16, 32]
LR_GRID = [1e-3, 3e-3, 1e-2]
TUNE_EPOCHS = 20  # fewer epochs for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_model(d, k, C, rng, vals_per_pos):
    # emb: list of (n_vals_j, k) for each position
    emb = [rng.normal(0, math.sqrt(2.0 / k), size=(len(v), k)).astype(np.float32) for v in vals_per_pos]
    W_out = rng.normal(0, math.sqrt(2.0 / (2 * k)), size=(2 * k, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    return emb, W_out, bias


def forward(emb, W_out, bias, X_idx, vals_per_pos):
    """X_idx: (n, d) indices into vals_per_pos. Returns scores (n, C)."""
    n, d = X_idx.shape
    k = emb[0].shape[1]
    C = bias.shape[0]
    z = np.zeros((n, k), np.float32)
    z2 = np.zeros((n, k), np.float32)
    for j in range(d):
        e = emb[j][X_idx[:, j]]  # (n, k)
        z += e
        z2 += e * e
    pairwise = (z * z - z2) * 0.5
    hidden = np.concatenate([z, pairwise], axis=1)  # (n, 2k)
    scores = hidden @ W_out + bias
    return scores


def loss_grad(scores, y):
    """Cross-entropy loss and gradient wrt scores. Returns (loss, grad_scores)."""
    n = scores.shape[0]
    scores_max = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad.astype(np.float32)


def backward(emb, W_out, X_idx, vals_per_pos, grad_scores):
    """Compute gradients for emb, W_out, bias."""
    n, d = X_idx.shape
    k = emb[0].shape[1]
    C = W_out.shape[1]
    # grad_hidden = grad_scores @ W_out.T  (n, 2k)
    grad_hidden = grad_scores @ W_out.T
    grad_z = grad_hidden[:, :k]
    grad_pairwise = grad_hidden[:, k:]
    # pairwise = (z^2 - z2)/2
    # d(pairwise)/dz = z, d(pairwise)/dz2 = -0.5
    z = np.zeros((n, k), np.float32)
    z2 = np.zeros((n, k), np.float32)
    for j in range(d):
        e = emb[j][X_idx[:, j]]
        z += e
        z2 += e * e
    grad_z += grad_pairwise * z
    grad_z2 = grad_pairwise * (-0.5)
    # Now grad_z and grad_z2 are gradients wrt z and z2
    # z = sum_j e_j, z2 = sum_j e_j^2
    # So for each position j: grad_e_j = grad_z + 2 * e_j * grad_z2
    grad_emb = [np.zeros_like(e) for e in emb]
    for j in range(d):
        e = emb[j][X_idx[:, j]]
        grad_e = grad_z + 2.0 * e * grad_z2
        # Accumulate into grad_emb[j]
        np.add.at(grad_emb[j], X_idx[:, j], grad_e)
    # W_out grad: hidden.T @ grad_scores
    hidden = np.concatenate([z, z2 * 0 + pairwise], axis=1)  # recompute or pass? pairwise = (z^2-z2)/2
    # Actually we need hidden = [z, pairwise]
    pairwise = (z * z - z2) * 0.5
    hidden = np.concatenate([z, pairwise], axis=1)
    grad_W_out = hidden.T @ grad_scores
    grad_bias = grad_scores.sum(axis=0)
    return grad_emb, grad_W_out.astype(np.float32), grad_bias.astype(np.float32)


def evaluate(emb, W_out, bias, X_idx, y, vals_per_pos):
    scores = forward(emb, W_out, bias, X_idx, vals_per_pos)
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def train_model(emb, W_out, bias, X_idx, y, X_val_idx, y_val, vals_per_pos, lr, epochs, rng):
    n = X_idx.shape[0]
    k = emb[0].shape[1]
    C = W_out.shape[1]
    # Adam state
    params = []
    for e in emb:
        params.append(e.ravel())
    params.append(W_out.ravel())
    params.append(bias.ravel())
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_idx[idx], y[idx]
            scores = forward(emb, W_out, bias, xb, vals_per_pos)
            loss, grad_scores = loss_grad(scores, yb)
            grad_emb, grad_W_out, grad_bias = backward(emb, W_out, xb, vals_per_pos, grad_scores)
            # Flatten grads
            grads = []
            for ge in grad_emb:
                grads.append(ge.ravel())
            grads.append(grad_W_out.ravel())
            grads.append(grad_bias.ravel())
            # L2 regularization
            for i, p in enumerate(params):
                grads[i] += L2 * p
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation
        val_acc = evaluate(emb, W_out, bias, X_val_idx, y_val, vals_per_pos)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    # Restore best
    if best_params:
        for i, p in enumerate(params):
            p[:] = best_params[i]
    return best_val


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]
    # Unique values per position
    vals_per_pos = [np.unique(X_bytes[:, j]) for j in range(d)]
    # Map bytes to indices
    X_idx = np.zeros((n, d), dtype=np.int32)
    for j in range(d):
        v = vals_per_pos[j]
        mapping = np.full(256, -1, dtype=np.int32)
        mapping[v] = np.arange(len(v))
        X_idx[:, j] = mapping[X_bytes[:, j]]
    # Split tuning (last 20%)
    split = int(n * 0.8)
    X_tr, X_val = X_idx[:split], X_idx[split:]
    y_tr, y_val = y[:split], y[split:]
    # Hyperparameter tuning
    rng = np.random.default_rng(SEED)
    best_k, best_lr, best_acc = 16, 3e-3, -1
    tune_epochs = min(TUNE_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for k in K_GRID:
        for lr in LR_GRID:
            emb, W_out, bias = make_model(d, k, C, rng, vals_per_pos)
            val_acc = train_model(emb, W_out, bias, X_tr, y_tr, X_val, y_val, vals_per_pos, lr, tune_epochs, rng)
            if val_acc > best_acc:
                best_acc, best_k, best_lr = val_acc, k, lr
    # Retrain on all data with best hyperparams
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W_out, bias = make_model(d, best_k, C, rng, vals_per_pos)
    train_model(emb, W_out, bias, X_idx, y, X_idx, y, vals_per_pos, best_lr, epochs, rng)
    # Export weights
    # Flatten emb to d x 256 x k (missing values zero)
    k = best_k
    emb_flat = np.zeros((d, 256, k), np.float32)
    for j in range(d):
        emb_flat[j, vals_per_pos[j]] = emb[j]
    # Pack
    magic = 0x464D3130  # 'FM10'
    hdr = struct.pack("<4i", magic, d, k, C)
    out = hdr + emb_flat.astype("<f4").tobytes() + W_out.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with pairwise interactions.
 * Weights: magic 0x464D3130, d, k, C; emb[d][256][k]; W_out[2k][C]; bias[C] (all float32 LE).
 * Input record: d bytes. Output: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * k + (long)2 * k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * k;
    const float *bias = W_out + (long)2 * k * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = calloc(k, sizeof(float));
    float *z2 = calloc(k, sizeof(float));
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        memset(z, 0, k * sizeof(float));
        memset(z2, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float v = e[f];
                z[f] += v;
                z2[f] += v * v;
            }
        }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int f = 0; f < k; f++) {
            float pairwise = (z[f] * z[f] - z2[f]) * 0.5f;
            for (int c = 0; c < C; c++) scores[c] += z[f] * W_out[(long)f * C + c];
            for (int c = 0; c < C; c++) scores[c] += pairwise * W_out[(long)(k + f) * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(z); free(z2); free(scores);
    return 0;
}
```