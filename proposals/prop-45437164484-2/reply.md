```hypothesis
Family: label-emblin (embedding + linear). Replaces label-linear's one-hot (4096-d) with per-position embeddings (16×E) followed by a linear layer. This low-rank factorization (16·256·E + 16·E·65 params) should generalize better on the small Tinyshakespeare training set (24k examples) while keeping inference fast (E≈8–16). We tune E on the last 20% of train.in, then retrain on all data. Failure: dev accuracy ≤ label-linear (0.3375) or inference cost ≥ label-gbdt.
```

```python
"""label-emblin: per-position embeddings + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the tuning split (last 20% of train.in)
EMBED_DIMS = [4, 8, 16, 32]
LRS = [1e-2, 3e-3, 1e-3]
L2S = [1e-4, 1e-5]
SEED = 1
BATCH = 256
MAX_EPOCHS = 200
PATIENCE = 10  # early stopping on validation loss


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, E, rng):
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W_out = (rng.normal(0, 0.01, (d * E, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    return emb, W_out, b_out


def forward(emb, W_out, b_out, X):
    # X: (N, d) uint8
    N, d = X.shape
    E = emb.shape[2]
    # gather embeddings: (N, d, E)
    # vectorized using advanced indexing
    idx = X[:, :, None] * np.ones((1, 1, E), dtype=np.int64)  # (N, d, E)
    # but we need to index emb[d, 256, E] -> (N, d, E)
    # simpler: loop over d (only 16)
    cat = np.empty((N, d * E), np.float32)
    for j in range(d):
        cat[:, j * E:(j + 1) * E] = emb[j, X[:, j]]
    scores = cat @ W_out + b_out  # (N, C)
    return scores, cat


def loss_grad(scores, y, cat, W_out, l2):
    N = scores.shape[0]
    # softmax + cross-entropy gradient
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N  # (N, C)
    # gradients
    grad_W_out = cat.T @ probs + l2 * W_out  # (d*E, C)
    grad_b_out = probs.sum(axis=0)           # (C,)
    # gradient wrt cat
    grad_cat = probs @ W_out.T               # (N, d*E)
    # gradient wrt embeddings
    d, E = grad_cat.shape[1] // W_out.shape[1], W_out.shape[1]  # not used directly
    d = grad_cat.shape[1] // E  # actually d = 16
    grad_emb = np.zeros_like(emb)
    for j in range(d):
        # grad_cat[:, j*E:(j+1)*E] is (N, E)
        # scatter-add to grad_emb[j, X[:, j]]
        np.add.at(grad_emb[j], X[:, j], grad_cat[:, j * E:(j + 1) * E])
    grad_emb += l2 * emb
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    return loss, grad_emb, grad_W_out, grad_b_out


def train_epoch(emb, W_out, b_out, X, y, l2, lr, batch, rng, adam_state):
    N = X.shape[0]
    order = rng.permutation(N)
    t = adam_state['t']
    for s in range(0, N, batch):
        idx = order[s:s + batch]
        xb, yb = X[idx], y[idx]
        scores, cat = forward(emb, W_out, b_out, xb)
        loss, g_emb, g_W, g_b = loss_grad(scores, yb, cat, W_out, l2)
        t += 1
        # Adam update for each param
        for param, grad, (m, v) in zip(
            [emb, W_out, b_out],
            [g_emb, g_W, g_b],
            adam_state['m_v']
        ):
            m *= 0.9; m += 0.1 * grad
            v *= 0.999; v += 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    adam_state['t'] = t
    return loss


def evaluate(emb, W_out, b_out, X, y):
    scores, _ = forward(emb, W_out, b_out, X)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def train_model(X_tr, y_tr, X_val, y_val, d, C, E, lr, l2, rng):
    emb, W_out, b_out = init_params(d, C, E, rng)
    # Adam state
    m_v = [
        (np.zeros_like(emb), np.zeros_like(emb)),
        (np.zeros_like(W_out), np.zeros_like(W_out)),
        (np.zeros_like(b_out), np.zeros_like(b_out))
    ]
    adam_state = {'t': 0, 'm_v': m_v}
    best_val_acc = -1.0
    best_params = None
    patience = 0
    for epoch in range(MAX_EPOCHS):
        train_epoch(emb, W_out, b_out, X_tr, y_tr, l2, lr, BATCH, rng, adam_state)
        val_acc = evaluate(emb, W_out, b_out, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (emb.copy(), W_out.copy(), b_out.copy())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_params is not None:
        emb, W_out, b_out = best_params
    return emb, W_out, b_out, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    n = X.shape[0]
    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_val_acc = -1.0
    best_cfg = None
    best_weights = None

    for E in EMBED_DIMS:
        for lr in LRS:
            for l2 in L2S:
                emb, W_out, b_out, val_acc = train_model(
                    X_tr, y_tr, X_val, y_val, d, C, E, lr, l2, rng
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_cfg = (E, lr, l2)
                    best_weights = (emb, W_out, b_out)

    # Retrain on full data with best config
    E, lr, l2 = best_cfg
    rng_full = np.random.default_rng(SEED)
    emb, W_out, b_out, _ = train_model(X, y, X_val, y_val, d, C, E, lr, l2, rng_full)

    # Write weights.bin
    # magic 0x454D424C ('LMBE' little-endian), d, E, C
    header = struct.pack("<4i", 0x454D424C, d, E, C)
    # embeddings: (d, 256, E) -> flatten row-major
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W_out.astype("<f4").tobytes()
    b_bytes = b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emblin inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + (long)d * 256 * E * 4 + (long)d * E * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *cat = malloc((long)d * E * sizeof(float));
    float *scores = malloc((long)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(cat + (long)j * E, src, E * sizeof(float));
        }

        // scores = cat @ W_out + b_out
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int fe = 0; fe < d * E; fe++) sum += cat[fe] * W_out[(long)fe * C + c];
            scores[c] += sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```