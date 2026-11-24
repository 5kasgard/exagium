```hypothesis
Family: label-embed (positional embeddings + optional ReLU layer).
Why: One-hot linear models ignore byte semantics; embeddings capture similarity and reduce dimensionality. A small hidden layer adds nonlinearity for sequential patterns. Tuning embedding/hidden sizes on the 20% tuning split should beat label-linear's 0.3375.
Failure: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2x.
```

```python
"""label-embed: positional embeddings + optional 1-hidden-layer ReLU MLP.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning (embedding size E, hidden size H)
GRID = [(4, 0), (4, 16), (4, 32), (4, 64),
        (8, 0), (8, 16), (8, 32), (8, 64),
        (16, 0), (16, 16), (16, 32), (16, 64)]
SEED = 1
LR, L2, BATCH, MAX_EPOCHS_TUNE, MAX_EPOCHS_FULL = 3e-3, 1e-4, 64, 30, 100
TUNE_FRAC = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def train_mlp(X, y, E, H, C, epochs, rng, lr=LR, l2=L2, batch=BATCH):
    """X: (n, d) uint8, y: (n,) int64. Returns (emb, W1, b1, W2, b2) or (emb, W, b) if H=0."""
    n, d = X.shape
    # Initialize embeddings: (d, 256, E)
    emb = rng.uniform(-0.1, 0.1, (d, 256, E)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (b, d) uint8 -> (b, d*E) float32
        bsz = X_batch.shape[0]
        Z = np.empty((bsz, d * E), np.float32)
        for j in range(d):
            Z[:, j*E:(j+1)*E] = emb[j, X_batch[:, j]]
        if H:
            a = Z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            return Z, a, h, logits
        else:
            logits = Z @ W + b
            return Z, logits

    def softmax_grad(logits, yy):
        # logits: (b, C), yy: (b,)
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            if H:
                Z, a, h, logits = forward(xb)
                g = softmax_grad(logits, yb)  # (b, C)
                # grad W2, b2
                grad_W2 = h.T @ g + l2 * W2
                grad_b2 = g.sum(0)
                # grad h
                gh = g @ W2.T
                # grad a (ReLU)
                ga = gh * (a > 0)
                # grad W1, b1
                grad_W1 = Z.T @ ga + l2 * W1
                grad_b1 = ga.sum(0)
                # grad Z
                gZ = ga @ W1.T
                # grad emb
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], gZ[:, j*E:(j+1)*E])
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                Z, logits = forward(xb)
                g = softmax_grad(logits, yb)
                grad_W = Z.T @ g + l2 * W
                grad_b = g.sum(0)
                gZ = g @ W.T
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], gZ[:, j*E:(j+1)*E])
                grads = [grad_emb, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params


def predict(X, params, H, C):
    n, d = X.shape
    if H:
        emb, W1, b1, W2, b2 = params
    else:
        emb, W, b = params
    E = emb.shape[2]
    Z = np.empty((n, d * E), np.float32)
    for j in range(d):
        Z[:, j*E:(j+1)*E] = emb[j, X[:, j]]
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z @ W + b
    return logits.argmax(axis=1).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = len(X_all)
    n_tune = int(n_all * TUNE_FRAC)
    n_train = n_all - n_tune
    X_train, X_tune = X_all[:n_train], X_all[n_train:]
    y_train, y_tune = y_all[:n_train], y_all[n_train:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E, best_H = 0, 0

    # Grid search on tuning split
    for E, H in GRID:
        print(f"Tuning E={E}, H={H}", file=sys.stderr)
        params = train_mlp(X_train, y_train, E, H, C, MAX_EPOCHS_TUNE, rng)
        y_pred = predict(X_tune, params, H, C)
        acc = accuracy(y_tune, y_pred)
        print(f"  tune acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_E, best_H = E, H

    print(f"Best: E={best_E}, H={best_H}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_mlp(X_all, y_all, best_E, best_H, C, MAX_EPOCHS_FULL, rng)

    # Pack weights.bin
    # Header: magic, d, E, H, C (all int32)
    magic = 0x454D4245  # "EMBE"
    header = struct.pack("<5i", magic, d, best_E, best_H, C)
    emb = final_params[0]
    out = header + emb.astype("<f4").tobytes()
    if best_H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input: 16-byte records. Output: one-byte class index per record. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }

    size_t emb_size = (size_t)d * 256 * E;
    size_t expected = 20 + emb_size * 4;
    if (H) {
        expected += (size_t)d * E * H * 4 + H * 4 + (size_t)H * C * 4 + C * 4;
    } else {
        expected += (size_t)d * E * C * 4 + C * 4;
    }
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_size;
    const float *b1 = W1 + (H ? (size_t)d * E * H : 0);
    const float *W2 = b1 + (H ? H : 0);
    const float *b2 = W2 + (H ? (size_t)H * C : 0);
    const float *W = W1, *b = b1; // for H=0

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *Z = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings into Z (flattened)
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(Z + j * E, e, E * sizeof(float));
        }

        if (H) {
            // hidden = ReLU(Z @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int fe = 0; fe < d * E; fe++) sum += Z[fe] * W1[(size_t)fe * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = Z @ W + b
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int fe = 0; fe < d * E; fe++) sum += Z[fe] * W[(size_t)fe * C + c];
                logits[c] = sum;
            }
        }

        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```