```hypothesis
Propose label-embnet: a two-hidden-layer embedding network that learns per-pixel embeddings summed into a first hidden layer (H1), then a second hidden layer (H2), then output. The ordinal pixels (0..16) are z-scored per feature, then linearly projected to H1, allowing non-linear interactions via two ReLU layers. This should capture more structure than label-linear while staying cheaper than label-mlp16 by using small H1,H2 (tuned on last 20% of train). If dev accuracy < 0.965 or inference cost > 7000 Ir/item, the design does not pay off.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameter grid searched on the tuning split (last 20% of train.in)
H1_CHOICES = [16, 32, 48]
H2_CHOICES = [8, 16, 24]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_size = int(n * tune_frac)
    idx = np.arange(n)
    # last 20% in file order
    tune_idx = idx[-tune_size:]
    train_idx = idx[:-tune_size]
    return X[train_idx], y[train_idx], X[tune_idx], y[tune_idx]


def build_encoding(card, X_train):
    """Return list of encoding matrices E_j (256 x enc_dim) and their dims."""
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X_train[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256,1)
            enc.append(E.astype(np.float32))
            dims.append(1)
        else:
            vals = np.unique(X_train[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            dims.append(len(vals))
    return enc, dims


def make_T(enc, dims, W1):
    """Compute lookup tables T_j[v] = E_j[v] @ W1_block for each feature."""
    T = []
    off = 0
    for E, d in zip(enc, dims):
        block = W1[off:off + d]  # (d, H1)
        T.append(E @ block)      # (256, H1)
        off += d
    return np.stack(T)  # (d, 256, H1)


def forward(Z, T, b1, W2, b2, W3, b3, H1, H2):
    """Z: (N, d) byte values. Returns logits (N, C)."""
    N, d = Z.shape
    # first layer: sum_j T[j, Z[:,j], :] + b1
    acc1 = np.broadcast_to(b1, (N, H1)).copy()
    for j in range(d):
        acc1 += T[j, Z[:, j]]
    h1 = np.maximum(acc1, 0)
    # second layer
    acc2 = h1 @ W2 + b2
    h2 = np.maximum(acc2, 0)
    # output
    logits = h2 @ W3 + b3
    return logits


def loss_grad(logits, y):
    """Cross-entropy loss and gradient w.r.t logits."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    probs[np.arange(N), y] -= 1.0
    return probs / N


def evaluate(Z, y, T, b1, W2, b2, W3, b3, H1, H2):
    logits = forward(Z, T, b1, W2, b2, W3, b3, H1, H2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(X_tr, y_tr, X_va, y_va, enc, dims, H1, H2, C, rng):
    d = X_tr.shape[1]
    k = sum(dims)  # total encoding dim = d for ordinal
    # parameters
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    T = make_T(enc, dims, W1)
    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Z, yy = X_tr[idx], y_tr[idx]
            # forward
            logits = forward(Z, T, b1, W2, b2, W3, b3, H1, H2)
            g = loss_grad(logits, yy)  # (B, C)
            # backward
            # output layer
            h2 = np.maximum((np.maximum(Z @ np.zeros((d, H1)) + b1, 0) @ W2 + b2), 0)  # dummy, recompute properly
            # Let's recompute activations for backward
            acc1 = np.broadcast_to(b1, (len(yy), H1)).copy()
            for j in range(d):
                acc1 += T[j, Z[:, j]]
            h1 = np.maximum(acc1, 0)
            acc2 = h1 @ W2 + b2
            h2 = np.maximum(acc2, 0)
            # gradients
            gW3 = h2.T @ g
            gb3 = g.sum(axis=0)
            gh2 = g @ W3.T * (acc2 > 0)
            gW2 = h1.T @ gh2
            gb2 = gh2.sum(axis=0)
            gh1 = gh2 @ W2.T * (acc1 > 0)
            # gh1 flows into each T[j, Z[:,j], :] and b1
            gb1 = gh1.sum(axis=0)
            # gradient for W1 via T
            gW1 = np.zeros_like(W1)
            off = 0
            for j in range(d):
                E = enc[j]
                dE = dims[j]
                # gh1[:, :] contributes to T[j, Z[:,j], :]
                # T[j] = E @ W1[off:off+dE]
                # dT/dW1_block = E^T
                # So gW1_block = E^T @ (gh1 indexed by Z[:,j])
                # We accumulate per feature
                idx_j = Z[:, j]
                # For each sample, the gradient for W1_block is E[idx_j]^T * gh1[sample]
                # Vectorized: (dE, 256) @ (256, H1) -> but we have per-sample gh1
                # Instead, we can do: for each feature value v, sum gh1 where Z[:,j]==v
                for v in range(256):
                    mask = (idx_j == v)
                    if mask.any():
                        gW1[off:off+dE] += E[v:v+1].T @ gh1[mask].sum(axis=0, keepdims=True)
                off += dE
            gW1 += L2 * W1
            gb1 += L2 * b1
            gW2 += L2 * W2
            gb2 += L2 * b2
            gW3 += L2 * W3
            gb3 += L2 * b3
            grads = [gW1, gb1, gW2, gb2, gW3, gb3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # update T after W1 change
            T = make_T(enc, dims, W1)
    # final T on full training stats (will be recomputed in main)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # split tuning set (last 20%)
    X_tr, y_tr, X_va, y_va = split_train_tune(X, y)

    # encoding on training split only
    enc, dims = build_encoding(card, X_tr)

    # grid search
    best_acc = -1
    best_params = None
    best_H1 = best_H2 = None
    for H1 in H1_CHOICES:
        for H2 in H2_CHOICES:
            if H2 > H1:
                continue
            params = train_model(X_tr, y_tr, X_va, y_va, enc, dims, H1, H2, C, rng)
            W1, b1, W2, b2, W3, b3 = params
            T = make_T(enc, dims, W1)
            acc = evaluate(X_va, y_va, T, b1, W2, b2, W3, b3, H1, H2)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_H1, best_H2 = H1, H2

    # retrain on full data with best H1,H2
    enc_full, dims_full = build_encoding(card, X)
    rng_full = np.random.default_rng(SEED)
    params = train_model(X, y, X, y, enc_full, dims_full, best_H1, best_H2, C, rng_full)
    W1, b1, W2, b2, W3, b3 = params
    T = make_T(enc_full, dims_full, W1)

    # pack weights: magic, d, H1, H2, C
    d = X.shape[1]
    magic = 0x454D424E  # 'EMBN'
    out = struct.pack("<5i", magic, d, best_H1, best_H2, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 20 || hd[0] != 0x454D424E) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        memcpy(acc2, b2, H2 * sizeof(float));
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu != 0.0f) {
                const float *w = W2 + (long)u * H2;
                for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
            }
        }
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        memcpy(sc, b3, C * sizeof(float));
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv != 0.0f) {
                const float *w = W3 + (long)v * C;
                for (int c = 0; c < C; c++) sc[c] += hv * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```