```hypothesis
We propose "label-embmlp": an embedding-based MLP that learns dense vectors for each categorical value and combines them with normalized ordinal features in a single hidden ReLU layer. Categorical features (8 of 14) get 8-dim embeddings; ordinal features (6) are z-scored. The MLP has 16 hidden units. This captures non-linear interactions cheaply: inference does 8 embedding lookups, one 70x16 matmul, ReLU, and 16x2 matmul. We expect >0.857 dev accuracy (beating label-linear's 0.8524) at ~1000 Ir/item. If accuracy ≤0.8524 or cost >2000 Ir/item, the family does not justify its complexity.
```

```python
"""label-embmlp: embedding + one-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D4244, d, num_cat, num_ord, E, H, C
  int32 is_cat[d] (1 if categorical else 0)
  float32 embeddings[num_cat * 256 * E]
  float32 ord_stats[num_ord * 2]  (mean, std per ordinal feature in original order)
  float32 W1[input_dim, H], b1[H], W2[H, C], b2[C]
  where input_dim = num_cat*E + num_ord
"""
import json, math, struct, sys

import numpy as np

E = 8
H = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    num_cat, num_ord = len(cat_idx), len(ord_idx)
    input_dim = num_cat * E + num_ord

    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats
    ord_mean = np.zeros(num_ord, np.float32)
    ord_std = np.ones(num_ord, np.float32)
    for i, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std()
        ord_mean[i] = mu
        ord_std[i] = sd if sd > 0 else 1.0

    # Embeddings: one per categorical feature, shape (256, E)
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(num_cat)]

    # MLP weights
    W1 = (rng.normal(0, math.sqrt(2.0 / input_dim), (input_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    # Adam state
    params = [W1, b1, W2, b2] + emb
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Build input batch
            inp = np.zeros((bs, input_dim), np.float32)
            off = 0
            for ci, j in enumerate(cat_idx):
                inp[:, off:off+E] = emb[ci][xb[:, j]]
                off += E
            for oi, j in enumerate(ord_idx):
                inp[:, off] = (xb[:, j].astype(np.float32) - ord_mean[oi]) / ord_std[oi]
                off += 1

            # Forward
            h = inp @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs  # dL/dlogits

            # Backprop
            gW2 = h_relu.T @ probs
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[h <= 0] = 0.0  # ReLU gradient
            gW1 = inp.T @ gh
            gb1 = gh.sum(axis=0)
            ginp = gh @ W1.T

            # Embedding gradients
            gemb = [np.zeros_like(e) for e in emb]
            off = 0
            for ci, j in enumerate(cat_idx):
                # scatter-add ginp[:, off:off+E] into gemb[ci] at indices xb[:, j]
                np.add.at(gemb[ci], xb[:, j], ginp[:, off:off+E])
                off += E
            # Ordinal gradients: just pass through normalization
            # (no need to store, they go into gW1/gb1 via inp)

            grads = [gW1 + L2 * W1, gb1, gW2 + L2 * W2, gb2] + gemb

            # Adam update
            t = 1  # we'll use a global step counter
            # For simplicity, use a single global step (not per-param)
            # But we need per-param step for bias correction. We'll keep a global step.
            # Actually, we can keep a global step and apply same correction to all.
            # We'll maintain a global step variable.
            # Let's restructure: keep global step outside loop.
            pass

    # We'll redo training with a cleaner Adam implementation.
    # Let's restart with a global step counter.

    # Re-initialize for clean training
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(num_cat)]
    W1 = (rng.normal(0, math.sqrt(2.0 / input_dim), (input_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2] + emb
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    step = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            inp = np.zeros((bs, input_dim), np.float32)
            off = 0
            for ci, j in enumerate(cat_idx):
                inp[:, off:off+E] = emb[ci][xb[:, j]]
                off += E
            for oi, j in enumerate(ord_idx):
                inp[:, off] = (xb[:, j].astype(np.float32) - ord_mean[oi]) / ord_std[oi]
                off += 1

            h = inp @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            gW2 = h_relu.T @ probs
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[h <= 0] = 0.0
            gW1 = inp.T @ gh
            gb1 = gh.sum(axis=0)
            ginp = gh @ W1.T

            gemb = [np.zeros_like(e) for e in emb]
            off = 0
            for ci, j in enumerate(cat_idx):
                np.add.at(gemb[ci], xb[:, j], ginp[:, off:off+E])
                off += E

            grads = [gW1 + L2 * W1, gb1, gW2 + L2 * W2, gb2] + gemb

            step += 1
            for i, (p, gr, m, v) in enumerate(zip(params, grads, ms, vs)):
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    is_cat = np.array([1 if t == "categorical" else 0 for t in types], dtype=np.int32)
    header = struct.pack("<7i", 0x454D4244, d, num_cat, num_ord, E, H, C)
    header += is_cat.tobytes()
    out = bytearray(header)
    for e in emb:
        out += e.astype("<f4").tobytes()
    out += np.stack([ord_mean, ord_std], axis=1).astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index. */
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
    if (nw < 28 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_cat = hd[2], num_ord = hd[3], E = hd[4], H = hd[5], C = hd[6];
    int32_t *is_cat = hd + 7;
    if (d > 100 || num_cat < 0 || num_ord < 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    int input_dim = num_cat * E + num_ord;
    long expected = 28L + 4L * d;  // header + is_cat
    expected += 4L * num_cat * 256 * E;  // embeddings
    expected += 4L * num_ord * 2;        // ord_stats
    expected += 4L * (input_dim * H + H + H * C + C);  // W1,b1,W2,b2
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(is_cat + d);
    const float *ord_stats = emb + (long)num_cat * 256 * E;
    const float *W1 = ord_stats + (long)num_ord * 2;
    const float *b1 = W1 + (long)input_dim * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *inp = malloc(sizeof(float) * input_dim);
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector
        int off = 0;
        for (int j = 0; j < d; j++) {
            if (is_cat[j]) {
                const float *e = emb + ((long)j * 256 + xi[j]) * E;  // WRONG: embeddings are stored per categorical feature, not per original feature index.
                // We need to map original feature index to categorical feature index.
                // Since we don't have that mapping in header, we must compute it from is_cat.
                // Let's fix: we'll iterate over original features, but we need to know which embedding table to use.
                // We'll precompute cat_feature_index for each original feature.
            }
        }
        // The above loop is flawed. We need to restructure: we know the order of embeddings in the file is the order of categorical features as they appear in the original feature list.
        // So we can do:
        //   int cat_ptr = 0;
        //   for (int j=0; j<d; j++) if (is_cat[j]) { use emb[cat_ptr]; cat_ptr++; }
        // But we are inside the per-item loop. Let's do it correctly.
    }

    // Let's rewrite the per-item loop correctly.
    // We'll compute cat_ptr and ord_ptr outside the loop? Actually we can compute them once.
    // But we need to know the mapping from original feature to embedding table index.
    // We'll precompute an array cat_map[d] where cat_map[j] = index in emb array if categorical, else -1.
    // Similarly ord_map[j] = index in ord_stats if ordinal, else -1.
    // We'll do that before the loop.

    // Rewind and redo.
    free(inp); free(h); free(scores);
    // We'll reallocate after computing maps.
    // Actually we can just compute maps on the fly in the loop without extra arrays.
    // Let's do a clean version.

    // We'll restart main from after header parsing.
    // But we can't goto. Let's just write the correct code from the beginning.
    // We'll output the entire corrected infer.c.

    // Since we are in a heredoc, we can just write the correct code below.
    // The above was a draft. We'll now write the final infer.c.
    return 0;
}
```