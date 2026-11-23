```hypothesis
Family: label-emblinear (embedding + linear). Each of the 14 features gets a small learned embedding
(ordinal: 256 bins, categorical: training categories + unseen). Concatenated embeddings fed to logistic
regression. Captures non-linear ordinal bin effects and categorical similarities at near-linear cost.
Why: Adult has 6 ordinal (binned) and 8 categorical features; linear model assumes monotonic ordinal
effects, but income vs age/hours is non-monotonic. Embeddings fix this with only ~4 dims/feature.
Failure: dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000 (vs 719 for label-linear).
```

```python
"""label-emblinear: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weight format: magic 0x454D424C, d, E, C, vocab_sizes[d], embeddings[d][vocab][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

EMBED_DIM = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-5, 256, 30


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]

    # vocab size per feature: ordinal=256, categorical=unique+1 (for 255=unseen)
    vocab_sizes = []
    for j, t in enumerate(types):
        if t == "ordinal":
            vocab_sizes.append(256)
        else:
            uniq = np.unique(X[:, j])
            vocab_sizes.append(len(uniq) + 1)  # +1 for unseen index 255

    rng = np.random.default_rng(SEED)
    # embeddings: list of (vocab, E) float32
    embeddings = [rng.normal(0, 0.01, (vocab_sizes[j], EMBED_DIM)).astype(np.float32) for j in range(d)]
    # linear: (d*E, C)
    W = rng.normal(0, math.sqrt(2.0 / (d * EMBED_DIM)), (d * EMBED_DIM, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = embeddings + [W, b]
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0

    n = X.shape[0]
    # use last 20% for early stopping / tuning (fixed split by file order)
    split = int(0.8 * n)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)

    def forward_idx(idxs):
        Xb = X[idxs]
        # concatenate embeddings
        emb_concat = np.zeros((len(idxs), d * EMBED_DIM), np.float32)
        for j in range(d):
            v = Xb[:, j]
            emb_concat[:, j * EMBED_DIM:(j + 1) * EMBED_DIM] = embeddings[j][v]
        logits = emb_concat @ W + b
        return logits, emb_concat

    def loss_grad(logits, yb):
        # softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        probs /= len(yb)
        return probs.astype(np.float32)

    best_va_acc = -1.0
    best_params = None
    for epoch in range(EPOCHS):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx = order[s:s + BATCH]
            logits, emb_concat = forward_idx(idx)
            g = loss_grad(logits, y[idx])
            # grads
            gW = emb_concat.T @ g + L2 * W
            gb = g.sum(axis=0)
            g_emb = [np.zeros_like(e) for e in embeddings]
            # backprop to embeddings
            g_emb_concat = g @ W.T  # (batch, d*E)
            for j in range(d):
                v = X[idx, j]
                gj = g_emb_concat[:, j * EMBED_DIM:(j + 1) * EMBED_DIM]
                np.add.at(g_emb[j], v, gj)
                g_emb[j] += L2 * embeddings[j]
            grads = g_emb + [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        if len(idx_va) > 0:
            logits_va, _ = forward_idx(idx_va)
            pred = logits_va.argmax(axis=1)
            va_acc = (pred == y[idx_va]).mean()
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_params = [p.copy() for p in params]

    # use best params
    if best_params is not None:
        embeddings, W, b = best_params[:-2], best_params[-2], best_params[-1]

    # retrain on full data for a few epochs (optional, but contract says train on all)
    # We'll just do a few more epochs on full data
    for epoch in range(5):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            logits, emb_concat = forward_idx(idx)
            g = loss_grad(logits, y[idx])
            gW = emb_concat.T @ g + L2 * W
            gb = g.sum(axis=0)
            g_emb = [np.zeros_like(e) for e in embeddings]
            g_emb_concat = g @ W.T
            for j in range(d):
                v = X[idx, j]
                gj = g_emb_concat[:, j * EMBED_DIM:(j + 1) * EMBED_DIM]
                np.add.at(g_emb[j], v, gj)
                g_emb[j] += L2 * embeddings[j]
            grads = g_emb + [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip([*embeddings, W, b], grads)):
                m, v_ = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    # magic, d, E, C
    out = struct.pack("<4i", 0x454D424C, d, EMBED_DIM, C)
    # vocab_sizes
    out += struct.pack(f"<{d}i", *vocab_sizes)
    # embeddings
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    # W, b
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emblinear inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index per record.
 * Weight format: magic 0x454D424C, d, E, C, vocab_sizes[d], embeddings[d][vocab][E], W[d*E][C], b[C] */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 14 || E != 4 || C != 2) { fprintf(stderr, "bad weights: dims\n"); return 3; }
    int32_t *vocab_sizes = hd + 4;
    long expected = 16 + 4L * d;  // header + vocab_sizes
    for (int j = 0; j < d; j++) {
        if (vocab_sizes[j] <= 0 || vocab_sizes[j] > 256) { fprintf(stderr, "bad vocab size\n"); return 3; }
        expected += (long)vocab_sizes[j] * E;
    }
    expected += (long)d * E * C;  // W
    expected += C;                // b
    expected *= 4;                // float32 bytes
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(vocab_sizes + d);
    const float *W = emb;
    for (int j = 0; j < d; j++) W += (long)vocab_sizes[j] * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // gather embeddings
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= vocab_sizes[j]) v = vocab_sizes[j] - 1; // clamp unseen
            const float *src = emb + ((long)vocab_sizes[j] * E * (j > 0)) + (long)v * E; // WRONG: need cumulative offset
        }
        // Fix: compute cumulative offsets
    }
    // Rewriting the embedding gather with correct offsets:
    free(concat); free(scores);
    // Actually, let's compute offsets array first.
    // We'll restart the loop properly.
    long *emb_offset = malloc(sizeof(long) * d);
    long off = 0;
    for (int j = 0; j < d; j++) {
        emb_offset[j] = off;
        off += (long)vocab_sizes[j] * E;
    }
    concat = malloc(sizeof(float) * d * E);
    scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= vocab_sizes[j]) v = vocab_sizes[j] - 1;
            const float *src = (const float *)emb + emb_offset[j] + (long)v * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }
        // linear: scores = concat @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int e = 0; e < d * E; e++) sum += concat[e] * W[(long)e * C + c];
            scores[c] += sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```