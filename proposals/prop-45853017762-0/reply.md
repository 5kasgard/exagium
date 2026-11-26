```hypothesis
Family: label-emb (embedding-based linear model). Each of the 14 byte features gets a learned embedding vector of dimension E (tuned on the last 20% of train.in). The item representation is the sum of its 14 embeddings, followed by a linear layer to 2 classes. This captures categorical similarities better than one-hot linear while keeping inference fast (14*E additions + E*2 multiply-adds). Expected to beat label-linear's 0.8524 dev accuracy with similar or lower CPU cost. Failure: accuracy ≤ 0.8524 or instructions/item ≥ label-linear.
```
```python
"""label-emb: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Tunes embedding dimension E on last 20% of train.in (file order), then trains on all.
Exports: magic 0x454D424C, d, E, C; float32 emb[d][256][E], W_out[E][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [4, 8, 16]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, emb, W_out, b_out, d, E, C):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s + 1024]
        yb = y[s:s + 1024]
        rep = np.zeros((len(xb), E), np.float32)
        for j in range(d):
            rep += emb[j, xb[:, j]]
        logits = rep @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_loop(X, y, emb, W_out, b_out, d, E, C, epochs, rng):
    params = [emb, W_out, b_out]
    shapes = [p.shape for p in params]
    flat = [p.ravel() for p in params]
    m = [np.zeros_like(f) for f in flat]
    v = [np.zeros_like(f) for f in flat]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(xb)
            rep = np.zeros((bsz, E), np.float32)
            for j in range(d):
                rep += emb[j, xb[:, j]]
            logits = rep @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz
            # gradients
            g_rep = probs @ W_out.T
            g_Wout = rep.T @ probs
            g_bout = probs.sum(axis=0)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_rep)
            grads_flat = [g_emb.ravel(), g_Wout.ravel(), g_bout.ravel()]
            for i in range(len(flat)):
                gr = grads_flat[i] + L2 * flat[i]
                t += 1
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                flat[i] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # reshape back (already in-place via flat views)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n_all = X_all.shape[0]
    split = int(0.8 * n_all)
    X_tr, X_tune = X_all[:split], X_all[split:]
    y_tr, y_tune = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)

    best_E = E_CANDIDATES[0]
    best_acc = -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))

    for E in E_CANDIDATES:
        emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
        W_out = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        train_loop(X_tr, y_tr, emb, W_out, b_out, d, E, C, epochs, rng)
        acc = evaluate(X_tune, y_tune, emb, W_out, b_out, d, E, C)
        if acc > best_acc:
            best_acc = acc
            best_E = E

    # retrain on all data with best_E
    E = best_E
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_out = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    train_loop(X_all, y_all, emb, W_out, b_out, d, E, C, epochs, rng)

    # write weights
    out = struct.pack("<4i", 0x454D424C, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: 1 byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W_out[E][C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d < 1 || E < 1 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rep = (float *)malloc(sizeof(float) * E);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) rep[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *row = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) rep[e] += row[e];
        }
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int e = 0; e < E; e++) {
            float re = rep[e];
            if (re == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += re * W_out[(long)e * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```