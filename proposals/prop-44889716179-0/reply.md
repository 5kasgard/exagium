```hypothesis
We propose label-fm, a factorization machine with a small latent dimension (k=2..8). FMs model all pairwise feature interactions via low-rank embeddings, which is crucial for tabular data like Adult where interactions (e.g., education × occupation) are predictive. Inference costs O(d·k) operations per item (d=14, k≤8), so it should be faster than label-mlp16 and competitive with label-linear. If dev accuracy does not exceed 0.8524 (label-linear) or latency exceeds label-linear by >2×, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the d=14 byte columns as a categorical field with 256 values (0-255).
Model: logit = w0 + Σ_i w[i,xi] + Σ_{i<j} ⟨v[i,xi], v[j,xj]⟩
Trains with Adam (lr=3e-3, L2=1e-4, batch=64) on 80% of train.in (file order),
tunes latent dimension k ∈ {2,4,8} on the held-out 20%, then retrains on all data.
Weights format: magic 0x464D0001, int32 d, k, C; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 50
PATIENCE = 3
K_CHOICES = [2, 4, 8]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(indices, w0, w, v, k):
    """indices: (B, d) uint8. Returns logits (B,)."""
    B, d = indices.shape
    # linear part
    linear = w0 + w[np.arange(d), indices].sum(axis=1)  # (B,)
    # interaction part via 0.5*( (Σ v)^2 - Σ v^2 )
    v_batch = v[np.arange(d), indices]  # (B, d, k)
    sum_v = v_batch.sum(axis=1)         # (B, k)
    sum_v_sq = (v_batch ** 2).sum(axis=1)  # (B, k)
    interaction = 0.5 * ((sum_v ** 2 - sum_v_sq).sum(axis=1))  # (B,)
    return linear + interaction

def train_fm(X, y, k, epochs, rng, X_val=None, y_val=None):
    """Returns (w0, w, v, best_epoch)."""
    n, d = X.shape
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.uniform(-0.01, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0
    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            B = len(yi)
            # forward
            logits = fm_forward(xi, w0, w, v, k)
            probs = sigmoid(logits)
            # gradient of BCE w.r.t logits
            grad_logits = (probs - yi).astype(np.float32) / B  # (B,)
            # gradients
            # linear: w0, w
            grad_w0 = grad_logits.sum()
            grad_w = np.zeros_like(w)
            np.add.at(grad_w, (np.arange(d)[:, None], xi.T), grad_logits)  # (d, 256)
            # interaction: v
            # Using efficient FM gradient: for each feature i, value xi, grad_v[i,xi] = grad_logits * (sum_v - v[i,xi])
            v_batch = v[np.arange(d), xi.T]  # (d, B, k)
            sum_v = v_batch.sum(axis=0)       # (B, k)
            grad_v = np.zeros_like(v)
            for i in range(d):
                vals = xi[:, i]
                # grad_v[i, vals] += grad_logits[:, None] * (sum_v - v_batch[i])
                diff = sum_v - v_batch[i]  # (B, k)
                np.add.at(grad_v[i], vals, grad_logits[:, None] * diff)
            # L2 regularization
            grad_w += L2 * w
            grad_v += L2 * v
            # Adam updates
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * grad_w0
            v_w0 = 0.999 * v_w0 + 0.001 * grad_w0 * grad_w0
            m_w0_hat = m_w0 / (1 - 0.9 ** t)
            v_w0_hat = v_w0 / (1 - 0.999 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            m_w_hat = m_w / (1 - 0.9 ** t)
            v_w_hat = v_w / (1 - 0.999 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            # v
            m_v = 0.9 * m_v + 0.1 * grad_v
            v_v = 0.999 * v_v + 0.001 * grad_v * grad_v
            m_v_hat = m_v / (1 - 0.9 ** t)
            v_v_hat = v_v / (1 - 0.999 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + 1e-8)
        # validation
        if X_val is not None:
            val_logits = fm_forward(X_val, w0, w, v, k)
            val_preds = (val_logits > 0).astype(np.int64)
            val_acc = (val_preds == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_params = (w0.copy(), w.copy(), v.copy())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
    if best_params is not None:
        w0, w, v = best_params
    return w0, w, v, best_epoch

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_k = K_CHOICES[0]
    best_val_acc = -1.0
    best_epoch = 0
    # tune k on validation split
    for k in K_CHOICES:
        w0, w, v, epoch = train_fm(X_train, y_train, k, MAX_EPOCHS_TUNE, rng, X_val, y_val)
        val_logits = fm_forward(X_val, w0, w, v, k)
        val_preds = (val_logits > 0).astype(np.int64)
        val_acc = (val_preds == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_epoch = epoch
    # retrain on full data with best_k for best_epoch epochs
    rng_full = np.random.default_rng(SEED)  # reset for reproducibility
    w0, w, v, _ = train_fm(X, y, best_k, best_epoch, rng_full)
    # write weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, best_k, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype(np.float32).tobytes()
    out += v.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class 0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float linear = w0;
        float sum_v[16] = {0};
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * k;
            for (int l = 0; l < k; l++) sum_v[l] += vj[l];
        }
        float interaction = 0.0f;
        for (int l = 0; l < k; l++) interaction += sum_v[l] * sum_v[l];
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *vj = v + ((long)j * 256 + val) * k;
            for (int l = 0; l < k; l++) interaction -= vj[l] * vj[l];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```