```hypothesis
Family: label-embedding. Dense embeddings (dim=4) for each categorical feature plus z-scored ordinal features, followed by a single linear layer. This gives richer categorical representations than one-hot linear while staying extremely fast (no hidden layer). It should beat label-linear (0.8524) on accuracy and be cheaper than label-mlp16 (2325 Ir/item). If dev accuracy ≤0.8524 or Ir/item ≥2000, the approach fails.
```
```python
"""label-embedding: categorical embeddings + ordinal z-score -> linear layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, numpy only.
Weights format: magic 0x454D4244, d, cat_count, ord_count, E, C, types[d], embeddings[cat_count][256][E], ord_mean[ord_count], ord_std[ord_count], W[cat_count*E+ord_count][C], b[C] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

E = 4
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_count, ord_count = len(cat_idx), len(ord_idx)
    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Normalize ordinal features once
    X_ord = ((X[:, ord_idx].astype(np.float32) - ord_mean) / ord_std).astype(np.float32)

    # Embeddings: one per categorical feature, size (256, E)
    emb = (rng.uniform(-0.01, 0.01, (cat_count, 256, E))).astype(np.float32)

    # Linear layer
    in_dim = cat_count * E + ord_count
    W = (rng.uniform(-0.01, 0.01, (in_dim, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    mEmb = [np.zeros_like(e) for e in emb]
    vEmb = [np.zeros_like(e) for e in emb]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = MAX_EPOCHS * steps_per_epoch
    t = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_cat = X[idx][:, cat_idx]          # (B, cat_count)
            x_ord = X_ord[idx]                  # (B, ord_count)
            yy = y[idx]

            # Embedding lookup
            B = len(idx)
            x_emb = np.empty((B, cat_count, E), np.float32)
            for c in range(cat_idx):
                x_emb[:, c] = emb[c, x_cat[:, c]]

            # Concatenate
            x = np.concatenate([x_emb.reshape(B, -1), x_ord], axis=1)  # (B, in_dim)

            # Forward
            logits = x @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = probs / B

            # Gradients
            grad_W = x.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_x = grad_logits @ W.T  # (B, in_dim)

            # Split grad_x into embedding part and ordinal part
            grad_emb_flat = grad_x[:, :cat_count * E].reshape(B, cat_count, E)
            # grad_ord = grad_x[:, cat_count * E:]  # not needed for parameters

            # Adam update for W, b
            t += 1
            for param, grad, m, v in [(W, grad_W, mW, vW), (b, grad_b, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # Adam update for each embedding table
            for c in range(cat_count):
                # scatter-add gradient for this categorical feature
                g = np.zeros((256, E), np.float32)
                np.add.at(g, x_cat[:, c], grad_emb_flat[:, c])
                g += L2 * emb[c]
                m = mEmb[c]; v = vEmb[c]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                emb[c] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    type_bytes = bytes([1 if t == "categorical" else 0 for t in types])
    header = struct.pack("<7I", 0x454D4244, d, cat_count, ord_count, E, C, 0)  # last 0 reserved
    out = header + type_bytes
    out += np.stack(emb).astype("<f4").tobytes()
    out += ord_mean.astype("<f4").tobytes()
    out += ord_std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: d bytes (categorical=category index, ordinal=quantized bin).
 * Output record: 1 byte class index (argmax). */
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
    if (nw < 32) { fprintf(stderr, "weights too small\n"); return 3; }

    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], cat_count = hd[2], ord_count = hd[3], E = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != cat_count + ord_count) { fprintf(stderr, "bad header dims\n"); return 3; }

    unsigned char *types = wb + 28;                 // 7*4 = 28
    float *emb = (float *)(types + d);              // cat_count * 256 * E
    float *ord_mean = emb + (long)cat_count * 256 * E;
    float *ord_std = ord_mean + ord_count;
    float *W = ord_std + ord_count;                 // (cat_count*E + ord_count) * C
    float *b = W + (long)(cat_count * E + ord_count) * C;

    // Expected weight size
    long expected = 28 + d + (long)cat_count * 256 * E * 4 + ord_count * 4 * 2
                    + (long)(cat_count * E + ord_count) * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * (cat_count * E + ord_count));
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int v_idx = 0;
        for (int j = 0; j < d; j++) {
            if (types[j]) { // categorical
                const float *e = emb + (long)j * 256 * E + xi[j] * E;
                for (int u = 0; u < E; u++) vec[v_idx++] = e[u];
            } else { // ordinal
                float val = (float)xi[j];
                vec[v_idx++] = (val - ord_mean[j]) / ord_std[j];
            }
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < v_idx; f++) {
            float xf = vec[f];
            for (int c = 0; c < C; c++) scores[c] += xf * W[(long)f * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```