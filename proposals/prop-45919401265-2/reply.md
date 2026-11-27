```hypothesis
We propose "label-embedding": each of the 16 context positions gets a learned 8-dimensional embedding per byte value; these are concatenated and fed to a linear classifier. This factorizes the 4096×65 weight matrix of label-linear into (16×256×8) + (128×65) ≈ 41k parameters, reducing overfitting and capturing byte similarities. It should beat label-linear's 0.3375 accuracy with similar or lower inference cost (embedding lookup + small matvec). Failure to beat label-linear would indicate the task needs full-rank position-byte interactions or that 8 dimensions are insufficient.
```

```python
"""label-embedding: per-position byte embeddings + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train)
SEED = 1
EMBED_DIM_CANDIDATES = [4, 8, 16]   # embedding dimension per position
LR = 0.01
BATCH = 64
EPOCHS_SEARCH = 20
EPOCHS_FULL = 30
L2 = 1e-5
VAL_FRACTION = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65

    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for validation (tuning)
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)

    def init_params(E):
        # embeddings: list of d arrays, each (256, E)
        emb = [rng.normal(0, 0.01, size=(256, E)).astype(np.float32) for _ in range(d)]
        # output weight: (d*E, C)
        W_out = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, C)).astype(np.float32)
        b_out = np.zeros(C, dtype=np.float32)
        return emb, W_out, b_out

    def forward(X_batch, emb, W_out, b_out):
        # X_batch: (batch, d)
        batch = X_batch.shape[0]
        # concatenate embeddings
        concat = np.empty((batch, d * emb[0].shape[1]), dtype=np.float32)
        for j in range(d):
            E = emb[j].shape[1]
            concat[:, j*E:(j+1)*E] = emb[j][X_batch[:, j]]
        scores = concat @ W_out + b_out
        return scores, concat

    def loss_and_grad(X_batch, y_batch, emb, W_out, b_out):
        batch = X_batch.shape[0]
        scores, concat = forward(X_batch, emb, W_out, b_out)
        # softmax cross-entropy
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        probs /= batch
        # gradients
        grad_W_out = concat.T @ probs + L2 * W_out
        grad_b_out = probs.sum(axis=0)
        # backprop to embeddings
        grad_concat = probs @ W_out.T
        grad_emb = [np.zeros_like(e) for e in emb]
        for j in range(d):
            E = emb[j].shape[1]
            g = grad_concat[:, j*E:(j+1)*E]   # (batch, E)
            # scatter-add: for each sample, add g[i] to grad_emb[j][X_batch[i, j]]
            np.add.at(grad_emb[j], X_batch[:, j], g)
            grad_emb[j] += L2 * emb[j]
        return grad_emb, grad_W_out, grad_b_out

    def evaluate(X_data, y_data, emb, W_out, b_out):
        scores, _ = forward(X_data, emb, W_out, b_out)
        preds = scores.argmax(axis=1)
        return (preds == y_data).mean()

    def train_model(X_tr, y_tr, X_val, y_val, E, epochs):
        emb, W_out, b_out = init_params(E)
        # Adam state
        m_emb = [np.zeros_like(e) for e in emb]
        v_emb = [np.zeros_like(e) for e in emb]
        m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
        m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n_tr = X_tr.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s+BATCH]
                X_batch, y_batch = X_tr[idx], y_tr[idx]
                grad_emb, grad_W, grad_b = loss_and_grad(X_batch, y_batch, emb, W_out, b_out)
                t += 1
                # update embeddings
                for j in range(d):
                    m_emb[j] = beta1 * m_emb[j] + (1 - beta1) * grad_emb[j]
                    v_emb[j] = beta2 * v_emb[j] + (1 - beta2) * grad_emb[j] * grad_emb[j]
                    m_hat = m_emb[j] / (1 - beta1 ** t)
                    v_hat = v_emb[j] / (1 - beta2 ** t)
                    emb[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)
                # update output layer
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
                m_hat = m_W / (1 - beta1 ** t)
                v_hat = v_W / (1 - beta2 ** t)
                W_out -= LR * m_hat / (np.sqrt(v_hat) + eps)

                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
                m_hat = m_b / (1 - beta1 ** t)
                v_hat = v_b / (1 - beta2 ** t)
                b_out -= LR * m_hat / (np.sqrt(v_hat) + eps)
        val_acc = evaluate(X_val, y_val, emb, W_out, b_out) if X_val is not None else 0.0
        return emb, W_out, b_out, val_acc

    # Hyperparameter search on validation split
    best_E, best_val = None, -1.0
    for E in EMBED_DIM_CANDIDATES:
        _, _, _, val_acc = train_model(X_train, y_train, X_val, y_val, E, EPOCHS_SEARCH)
        if val_acc > best_val:
            best_val, best_E = val_acc, E
    print(f"Chosen E={best_E} with val acc={best_val:.4f}", file=sys.stderr)

    # Retrain on full dataset
    emb, W_out, b_out, _ = train_model(X, y, None, None, best_E, EPOCHS_FULL)

    # Save weights
    # Format: magic(4), d(4), E(4), C(4) all int32 little-endian
    # then embeddings: d * 256 * E float32 (position-major: pos 0..d-1, byte 0..255, E dims)
    # then W_out: (d*E) * C float32
    # then b_out: C float32
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<4i", magic, d, best_E, C)
    # flatten embeddings in order
    emb_flat = np.concatenate([e.ravel() for e in emb]).astype("<f4")
    W_flat = W_out.astype("<f4").ravel()
    b_flat = b_out.astype("<f4")
    out = header + emb_flat.tobytes() + W_flat.tobytes() + b_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + (long)d * 256 * E * 4
        + (long)d * E * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // init scores with bias
        for (int c = 0; c < C; c++) scores[c] = b_out[c];

        // accumulate embedding contributions
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            const float *w = W_out + (long)j * E * C;
            for (int e_idx = 0; e_idx < E; e_idx++) {
                float val = e[e_idx];
                const float *w_row = w + (long)e_idx * C;
                for (int c = 0; c < C; c++) scores[c] += val * w_row[c];
            }
        }

        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```