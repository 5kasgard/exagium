```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Each of the 16 context bytes is mapped to a learned dense vector (embedding size E). The embeddings are summed, passed through a ReLU hidden layer of size H, then a linear output layer to 65 classes. This factorizes the large one-hot weight matrix (16*256*65) into smaller embedding tables, improving generalization and reducing parameters. Tuning E and H on the 20% hold-out should beat label-linear's 0.3375 accuracy at comparable or lower inference cost (embedding lookup + small matvecs). If dev accuracy does not exceed 0.34 or inference cost exceeds label-linear by >2x, the approach fails.
```

```python
"""label-embmlp: embedding + one-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains on last 20% of train.in for hyperparameter selection (E, H), then on all train.
Weights format: magic 0x454D4231, d, E, H, C; float32 embeddings[d][256][E], b_h[E], W_h[E][H], b_out[C], W_out[H][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Hyperparameter grid for tuning split
E_GRID = [16, 32]
H_GRID = [16, 32, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y):
    d, E, H, C = params['d'], params['E'], params['H'], params['C']
    emb, b_h, W_h, b_out, W_out = params['emb'], params['b_h'], params['W_h'], params['b_out'], params['W_out']
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = slice(s, min(s + BATCH, n))
        xb = X[idx]
        # sum embeddings
        sum_emb = np.zeros((xb.shape[0], E), np.float32)
        for j in range(d):
            sum_emb += emb[j][xb[:, j]]
        # hidden
        h = sum_emb @ W_h + b_h
        h = np.maximum(h, 0)
        # output
        scores = h @ W_out + b_out
        pred = scores.argmax(axis=1)
        correct += (pred == y[idx]).sum()
    return correct / n


def train_config(X_tr, y_tr, X_val, y_val, d, C, E, H, rng):
    # Initialize parameters
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    W_h = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    # Adam state
    params = [emb, [W_h], [b_h], [W_out], [b_out]]
    # Flatten for Adam
    flat = []
    for group in params:
        for p in group:
            flat.append(p)
    m = [np.zeros_like(p) for p in flat]
    v = [np.zeros_like(p) for p in flat]
    t = 0

    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_tr[idx]
            yb = y_tr[idx]
            bs = len(yb)

            # Forward
            sum_emb = np.zeros((bs, E), np.float32)
            for j in range(d):
                sum_emb += emb[j][xb[:, j]]
            h_pre = sum_emb @ W_h + b_h
            h = np.maximum(h_pre, 0)
            scores = h @ W_out + b_out
            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            dW_out = h.T @ probs + L2 * W_out
            db_out = probs.sum(axis=0)
            dh = probs @ W_out.T
            dh *= (h_pre > 0)
            dW_h = sum_emb.T @ dh + L2 * W_h
            db_h = dh.sum(axis=0)
            dsum_emb = dh @ W_h.T  # (bs, E)

            # Embedding gradients
            demb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dsum_emb)

            # Collect gradients in same order as flat
            grads = []
            for j in range(d):
                grads.append(demb[j])
            grads.append(dW_h)
            grads.append(db_h)
            grads.append(dW_out)
            grads.append(db_out)

            t += 1
            for i, (p, gr) in enumerate(zip(flat, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Reassemble params
    params_dict = {
        'd': d, 'E': E, 'H': H, 'C': C,
        'emb': emb, 'W_h': W_h, 'b_h': b_h, 'W_out': W_out, 'b_out': b_out
    }
    val_acc = accuracy(params_dict, X_val, y_val)
    return val_acc, params_dict


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_cfg = None

    for E in E_GRID:
        for H in H_GRID:
            # Use a deterministic sub-seed for each config
            cfg_rng = np.random.default_rng(SEED + E * 1000 + H)
            acc, params = train_config(X_tr, y_tr, X_val, y_val, d, C, E, H, cfg_rng)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_cfg = (E, H)
            print(f"E={E} H={H} val_acc={acc:.4f}", file=sys.stderr)

    print(f"Best config: E={best_cfg[0]} H={best_cfg[1]} val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    full_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_config(X, y, X_val, y_val, d, C, best_cfg[0], best_cfg[1], full_rng)

    # Write weights.bin
    emb = final_params['emb']
    W_h = final_params['W_h']
    b_h = final_params['b_h']
    W_out = final_params['W_out']
    b_out = final_params['b_out']
    E, H = best_cfg

    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in emb)
    # b_h, W_h, b_out, W_out
    bh_bytes = b_h.astype("<f4").tobytes()
    Wh_bytes = W_h.astype("<f4").tobytes()
    bout_bytes = b_out.astype("<f4").tobytes()
    Wout_bytes = W_out.astype("<f4").tobytes()

    out = header + emb_bytes + bh_bytes + Wh_bytes + bout_bytes + Wout_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights format: magic 0x454D4231, d, E, H, C (int32 little-endian)
 * float32 embeddings[d][256][E], b_h[E], W_h[E][H], b_out[C], W_out[H][C] */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + E + (long)E * H + C + (long)H * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *b_h = emb + (long)d * 256 * E;
    const float *W_h = b_h + E;
    const float *b_out = W_h + (long)E * H;
    const float *W_out = b_out + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // sum embeddings
        for (int e = 0; e < E; e++) sum_emb[e] = b_h[e];
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += eptr[e];
        }

        // hidden layer: h = relu(sum_emb @ W_h + b_h) but b_h already added
        for (int u = 0; u < H; u++) {
            float acc = 0.0f;
            for (int e = 0; e < E; e++) acc += sum_emb[e] * W_h[(long)e * H + u];
            h[u] = acc > 0.0f ? acc : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float acc = b_out[c];
            for (int u = 0; u < H; u++) acc += h[u] * W_out[(long)u * C + c];
            scores[c] = acc;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```