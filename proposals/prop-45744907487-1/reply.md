```hypothesis
Family: label-mlp8 (small one-hidden-layer MLP with 8 ReLU units).
Why: The optdigits task has 64 ordinal pixel-count features; a tiny non-linear layer can capture stroke interactions that linear models miss, while staying far cheaper than the 16-unit MLP (label-mlp16). Training uses the same per-byte lookup tables, so inference is a single matrix-vector product plus ReLU.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or latency ≥ label-mlp16, the extra non-linearity does not justify its cost.
```

```python
"""label-mlp8: 8-unit ReLU MLP with per-byte lookup tables.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all data."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8
SEED = 1
# Grid for tuning split
LR_GRID = [0.003, 0.01, 0.03]
L2_GRID = [1e-4, 1e-3]
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5  # early stopping on validation loss


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def build_table(enc, W1, b1):
    """Precompute per-byte contributions to hidden layer: T[j][byte][h]."""
    d = len(enc)
    W = W1.shape[1]
    T = np.zeros((d, 256, W), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        T[j] = (E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32)
        off += E.shape[1]
    return T


def train_step(z, y, W1, b1, W2, b2, LR, L2, state, t):
    """One minibatch Adam step, returns updated params and state."""
    n = z.shape[0]
    # forward
    a = z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    # softmax + cross-entropy gradient
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    grad_logits = (probs / n).astype(np.float32)
    # backprop
    grad_W2 = h.T @ grad_logits + L2 * W2
    grad_b2 = grad_logits.sum(axis=0)
    grad_h = grad_logits @ W2.T
    grad_a = grad_h * (a > 0)
    grad_W1 = z.T @ grad_a + L2 * W1
    grad_b1 = grad_a.sum(axis=0)
    grads = [grad_W1, grad_b1, grad_W2, grad_b2]
    params = [W1, b1, W2, b2]
    # Adam update
    for i, (p, gr) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= 0.9; m += 0.1 * gr
        v *= 0.999; v += 0.001 * gr * gr
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params, state


def evaluate(Z, y, W1, b1, W2, b2):
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    types = card["input"]["types"]

    # encoding (same for all hyperparameters)
    enc = encode_features(X, types)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # split: last 20% for tuning
    n_valid = n // 5
    train_idx = np.arange(n - n_valid)
    valid_idx = np.arange(n - n_valid, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_valid, y_valid = Z[valid_idx], y[valid_idx]

    best_acc = -1.0
    best_params = None
    best_hyper = None

    for lr in LR_GRID:
        for l2 in L2_GRID:
            # deterministic init per hyperparameter combo
            rng = np.random.default_rng(SEED + hash((lr, l2)) % 2**32)
            W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
            b1 = np.zeros(H, np.float32)
            W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
            b2 = np.zeros(C, np.float32)
            params = [W1, b1, W2, b2]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

            t = 0
            best_valid_loss = float('inf')
            patience = 0
            best_local_params = None

            for epoch in range(MAX_EPOCHS):
                order = rng.permutation(len(Z_train))
                for s in range(0, len(Z_train), BATCH):
                    idx = order[s:s + BATCH]
                    z, yy = Z_train[idx], y_train[idx]
                    t += 1
                    params, state = train_step(z, yy, *params, lr, l2, state, t)

                # validation loss (cross-entropy)
                a = Z_valid @ params[0] + params[1]
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                valid_loss = -np.log(probs[np.arange(len(y_valid)), y_valid] + 1e-12).mean()

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience = 0
                    best_local_params = [p.copy() for p in params]
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break

            # evaluate best epoch on validation
            if best_local_params is not None:
                acc = evaluate(Z_valid, y_valid, *best_local_params)
                if acc > best_acc:
                    best_acc = acc
                    best_params = best_local_params
                    best_hyper = (lr, l2)

    print(f"Best hyperparams: lr={best_hyper[0]}, l2={best_hyper[1]}, valid_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best hyperparams
    lr, l2 = best_hyper
    rng = np.random.default_rng(SEED)
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            t += 1
            params, state = train_step(z, yy, *params, lr, l2, state, t)

    W1, b1, W2, b2 = params

    # Build lookup tables
    T = build_table(enc, W1, b1)

    # Export: magic, d, H, C; then T (d*256*H float32), b1 (H), W2 (H*C), b2 (C)
    magic = 0x54424C31  # "TBL1"
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp8 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output: 1 byte (class 0-9).
 * Weights format: magic(0x54424C31), d, H, C; T[d][256][H] float32; b1[H]; W2[H][C]; b2[C]. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H != 8) { fprintf(stderr, "unexpected H=%d\n", H); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) scores[c] += hv * W2[(long)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```