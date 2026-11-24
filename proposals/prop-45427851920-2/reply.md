```hypothesis
Family: label-shared-embed — a one-hidden-layer MLP where the 16 input positions share a single byte embedding table (256 × E) followed by position-specific projections (E → H). This reduces parameters (256·E + 16·E·H vs 16·256·H) and may generalize better on 24k examples. Inference cost matches a per-position MLP because we pre-compute the 16×256×H lookup tables. If dev accuracy does not exceed label-linear (0.3375), the shared embedding constraint is too restrictive for this task.
```

```python
"""label-shared-embed: shared byte embedding + position-specific projection + one ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hyperparameters (E, H, lr, epochs) chosen on last 20% of train.in (tuning split), then trained on all train."""
import json, math, struct, sys

import numpy as np

# hyperparameter grid searched on tuning split
E_GRID = [16, 32]
H_GRID = [32, 64, 128]
LR_GRID = [1e-3, 3e-3]
SEED = 1
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
L2 = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y, C):
    E, P, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        idx = slice(s, min(s + 1024, n))
        xb, yb = X[idx], y[idx]
        emb = E[xb]                     # (b, d, E)
        proj = np.einsum('bde,deh->bdh', emb, P)  # (b, d, H)
        h = proj.sum(axis=1) + b1       # (b, H)
        h = np.maximum(h, 0)
        logits = h @ W2 + b2            # (b, C)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X_tr, y_tr, X_val, y_val, C, E, H, lr, rng):
    d = X_tr.shape[1]
    # initialize
    E_emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    P_proj = (rng.normal(0, 0.01, (d, E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E_emb, P_proj, b1, W2, b2]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_val = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            b = xb.shape[0]

            # forward
            emb = E_emb[xb]                     # (b, d, E)
            proj = np.einsum('bde,deh->bdh', emb, P_proj)  # (b, d, H)
            h_pre = proj.sum(axis=1) + b1       # (b, H)
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2                # (b, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(b), yb] -= 1.0
            probs /= b

            # gradients
            g_W2 = h.T @ probs + L2 * W2
            g_b2 = probs.sum(axis=0)
            g_h = probs @ W2.T
            g_h[h_pre <= 0] = 0.0
            g_b1 = g_h.sum(axis=0)
            # g_proj: (b, d, H)
            g_proj = np.zeros((b, d, H), np.float32)
            for j in range(d):
                g_proj[:, j, :] = g_h
            # g_P: (d, E, H)
            g_P = np.einsum('bde,bdh->deh', emb, g_proj) + L2 * P_proj
            # g_emb: (b, d, E)
            g_emb = np.einsum('bdh,deh->bde', g_proj, P_proj)
            # accumulate g_E: (256, E)
            g_E = np.zeros_like(E_emb)
            np.add.at(g_E, xb.reshape(-1), g_emb.reshape(-1, E))

            grads = [g_E, g_P, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        val_acc = accuracy(params, X_val, y_val, C)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]

    return best_params, best_val


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # tuning split = last 20% in file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_overall = None
    best_score = -1.0
    best_cfg = None

    for E in E_GRID:
        for H in H_GRID:
            for lr in LR_GRID:
                params, val_acc = train_model(X_tr, y_tr, X_val, y_val, C, E, H, lr, rng)
                if val_acc > best_score:
                    best_score = val_acc
                    best_overall = params
                    best_cfg = (E, H, lr)

    # retrain on full data with best config
    E, H, lr = best_cfg
    final_params, _ = train_model(X, y, X_val, y_val, C, E, H, lr, rng)
    E_emb, P_proj, b1, W2, b2 = final_params

    # build lookup tables T[d][256][H] = E_emb[byte] @ P_proj[pos]
    T = np.einsum('ve,deh->dvh', E_emb, P_proj).astype(np.float32)  # (d, 256, H)

    # write weights: magic, d, H, C; T (d*256*H floats), b1 (H), W2 (H*C), b2 (C)
    magic = 0x54424C31
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
/* label-shared-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x54424C31, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

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
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```