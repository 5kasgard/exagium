```hypothesis
Family: label-emb (embedding-based linear model). Treats all 14 byte features as categorical, learns a small embedding per feature value (dim E tuned on 20% tuning split), then a linear layer on concatenated embeddings. Compressed to per-feature lookup tables for fast inference (same cost as label-linear). Expects to beat label-linear (0.8524) by capturing interactions via embeddings; if dev accuracy <= 0.8524, the approach fails.
```

```python
"""label-emb: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains embeddings (dim E) for each of 256 feature values, then linear layer.
Chooses E on last 20% of train.in (file order), then trains on all data.
Exports as per-feature lookup tables: magic 0x454D424C, d, H=0, C; float32 T[d][256][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_E = [2, 4, 8, 16]  # embedding dimensions to try
MAGIC = 0x454D424C  # 'EMBL' in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_val, y_val, d, C, E, rng):
    n_tr = X_tr.shape[0]
    # embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # linear weights: (d*E, C)
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        B = X_batch.shape[0]
        scores = np.zeros((B, C), np.float32)
        for j in range(d):
            idx = X_batch[:, j]
            e = emb[j, idx]  # (B, E)
            w = W[j * E:(j + 1) * E]  # (E, C)
            scores += e @ w
        scores += b
        return scores

    def loss_grad(scores, y_batch):
        B = scores.shape[0]
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B  # (B, C)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            scores = forward(Xb)
            g = loss_grad(scores, yb)  # (B, C)

            # gradients
            g_emb = np.zeros_like(emb)
            g_W = np.zeros_like(W)
            g_b = g.sum(axis=0)

            for j in range(d):
                idx_j = Xb[:, j]  # (B,)
                e = emb[j, idx_j]  # (B, E)
                w = W[j * E:(j + 1) * E]  # (E, C)
                # dL/demb = g @ w.T  (B, E)
                g_emb[j, idx_j] += g @ w.T
                # dL/dW = e.T @ g  (E, C)
                g_W[j * E:(j + 1) * E] += e.T @ g

            g_emb += L2 * emb
            g_W += L2 * W
            grads = [g_emb, g_W, g_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    val_scores = forward(X_val)
    val_pred = val_scores.argmax(axis=1)
    acc = (val_pred == y_val).mean()
    return acc, emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    tr_idx, val_idx = idx[:split], idx[split:]

    rng = np.random.default_rng(SEED)
    best_E, best_acc = CANDIDATE_E[0], -1.0
    for E in CANDIDATE_E:
        # new rng for each E to keep deterministic across runs
        rng_E = np.random.default_rng(SEED + E)
        acc, _, _, _ = train_eval(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], d, C, E, rng_E)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # final training on all data
    rng_final = np.random.default_rng(SEED + best_E)
    _, emb, W, b = train_eval(X, y, X[:1], y[:1], d, C, best_E, rng_final)

    # build lookup tables T[j][v] = emb[j,v] @ W[j*E:(j+1)*E]  (C,)
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        T[j] = emb[j] @ W[j * best_E:(j + 1) * best_E]

    # write weights.bin
    header = struct.pack("<4i", MAGIC, d, 0, C)
    body = T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index.
 * Weights format: magic 0x454D424C, int32 d, H=0, C; float32 T[d][256][C]; float32 b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```