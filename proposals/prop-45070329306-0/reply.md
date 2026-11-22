```hypothesis
Family: label-emb-mlp (embedding MLP). 
Why: Adult has 9 categorical columns with many levels; one-hot (label-linear) is wide but shallow. 
Learned embeddings (dim=8) compress categorical information, then a 16-unit ReLU layer captures interactions. 
This should beat label-linear's 0.8524 accuracy while staying fast (small dense matvecs). 
Failure signal: dev accuracy ≤ 0.8524 or instructions/item ≥ label-linear's 719.
```

```python
"""label-emb-mlp: embedding + one hidden ReLU layer for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, H, C, n_cat, n_ord
  float32 ord_mean[n_ord], ord_std[n_ord]
  float32 emb[n_cat][256][E]   (categorical columns in file order)
  float32 W1[input_dim][H], b1[H]
  float32 W2[H][C], b2[C]
where input_dim = n_cat*E + n_ord, E=8, H=16, C=2."""
import json, math, struct, sys
import numpy as np

E, H = 8, 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    # Identify categorical / ordinal columns
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)
    input_dim = n_cat * E + n_ord

    # Ordinal statistics on full train
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std  = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Embedding tables: one per categorical column, size 256 x E
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 0.01, (n_cat, 256, E))).astype(np.float32)

    # Hidden / output weights
    W1 = (rng.normal(0, math.sqrt(2.0 / input_dim), (input_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Build input vectors
            z = np.empty((bs, input_dim), np.float32)
            # ordinal part
            z[:, :n_ord] = ((xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)
            # categorical embeddings
            off = n_ord
            for ci, cj in enumerate(cat_idx):
                z[:, off:off+E] = emb[ci, xb[:, cj]]
                off += E

            # Forward
            h_pre = z @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backprop
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[h_pre <= 0] = 0.0
            gW1 = z.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            gz = gh @ W1.T

            # Embedding gradients
            gemb = np.zeros_like(emb)
            off = n_ord
            for ci, cj in enumerate(cat_idx):
                np.add.at(gemb[ci], xb[:, cj], gz[:, off:off+E])
                off += E

            grads = [gemb, gW1, gb1, gW2, gb2]

            # Adam update
            t = 1  # simplified: constant step count per batch
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<6i", magic, d, E, H, C, n_cat, n_ord)
    body = bytearray()
    body += ord_mean.astype("<f4").tobytes()
    body += ord_std.astype("<f4").tobytes()
    body += emb.astype("<f4").tobytes()
    body += W1.astype("<f4").tobytes()
    body += b1.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes()
    body += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4], n_cat = hd[5], n_ord = hd[6];
    if (C != 2 || d != 14 || n_cat + n_ord != d) { fprintf(stderr, "bad header dims\n"); return 3; }
    int input_dim = n_cat * E + n_ord;
    size_t expected = 24
        + 4L * n_ord * 2                     // ord_mean, ord_std
        + 4L * n_cat * 256 * E               // emb
        + 4L * input_dim * H + 4L * H        // W1, b1
        + 4L * H * C + 4L * C;               // W2, b2
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    float *p = (float *)(hd + 6);
    float *ord_mean = p; p += n_ord;
    float *ord_std  = p; p += n_ord;
    float *emb = p; p += n_cat * 256 * E;
    float *W1 = p; p += input_dim * H;
    float *b1 = p; p += H;
    float *W2 = p; p += H * C;
    float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * input_dim);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // ordinal part
        for (int j = 0; j < n_ord; j++) {
            z[j] = (xi[j] - ord_mean[j]) / ord_std[j];
        }
        // categorical embeddings
        int off = n_ord;
        for (int ci = 0, cj = 0; cj < d; cj++) {
            if (cj == 0 || cj == 2 || cj == 4 || cj == 10 || cj == 11 || cj == 12) continue; // ordinal indices
            const float *e = emb + (long)ci * 256 * E + xi[cj] * E;
            for (int k = 0; k < E; k++) z[off++] = e[k];
            ci++;
        }

        // hidden layer
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < input_dim; v++) sum += z[v] * W1[v * H + u];
            h[u] = sum > 0 ? sum : 0;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            logits[c] = sum;
        }

        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```