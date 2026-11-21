```hypothesis
Family: label-emb (embedding + shallow MLP). Categorical features get learned 8-d embeddings; ordinal features are z-scored scalars. Concatenated vector (8*8 + 6 = 70) feeds a 8-unit ReLU hidden layer then 2-class output. This captures non-linear feature interactions better than linear models while keeping inference fast (small dense matmuls). If dev accuracy ≤ 0.855 or Ir/item > 1000, the approach fails.
```
```python
"""label-emb: embeddings for categorical, z-score for ordinal, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, H, C
  uint8 type[d] (0=ordinal,1=categorical)
  For each feature:
    if categorical: float32[256*E] embedding table
    if ordinal: float32[2] mean, std
  float32[W*C] W_out, float32[C] b_out   (W = H if H>0 else input_dim)
  if H>0: float32[input_dim*H] W_hid, float32[H] b_hid
"""
import json, math, struct, sys

import numpy as np

E = 8          # embedding size
H = 8          # hidden units (0 = linear)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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
    n_cat, n_ord = len(cat_idx), len(ord_idx)
    input_dim = n_cat * E + n_ord

    # ordinal statistics
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.ones(n_ord, np.float32)
    for k, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[k] = v.mean()
        ord_std[k] = v.std() or 1.0

    # embeddings init
    rng = np.random.default_rng(SEED)
    emb = rng.uniform(-1, 1, (n_cat, 256, E)).astype(np.float32) * math.sqrt(6.0 / E)

    # hidden layer
    if H:
        W_hid = rng.uniform(-1, 1, (input_dim, H)).astype(np.float32) * math.sqrt(6.0 / input_dim)
        b_hid = np.zeros(H, np.float32)
    else:
        W_hid = np.zeros((input_dim, 0), np.float32)
        b_hid = np.zeros(0, np.float32)

    # output layer
    W_out = rng.uniform(-1, 1, (H if H else input_dim, C)).astype(np.float32) * math.sqrt(6.0 / (H if H else input_dim))
    b_out = np.zeros(C, np.float32)

    params = [emb, W_hid, b_hid, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yy = y[idx]

            # build input batch
            batch_in = np.zeros((len(idx), input_dim), np.float32)
            for k, j in enumerate(cat_idx):
                batch_in[:, k*E:(k+1)*E] = emb[k, xb[:, j]]
            for k, j in enumerate(ord_idx):
                batch_in[:, n_cat*E + k] = (xb[:, j] - ord_mean[k]) / ord_std[k]

            if H:
                hid = batch_in @ W_hid + b_hid
                hid = np.maximum(hid, 0)
                logits = hid @ W_out + b_out
                g_out = softmax_grad(logits, yy)
                g_hid = (g_out @ W_out.T) * (hid > 0)
                grad_W_out = hid.T @ g_out + L2 * W_out
                grad_b_out = g_out.sum(0)
                grad_W_hid = batch_in.T @ g_hid + L2 * W_hid
                grad_b_hid = g_hid.sum(0)
                # embedding grads
                grad_emb = np.zeros_like(emb)
                for k in range(n_cat):
                    for b_idx, row_idx in enumerate(idx):
                        grad_emb[k, xb[b_idx, cat_idx[k]]] += g_hid[b_idx, k*E:(k+1)*E]
                grad_emb += L2 * emb
                grads = [grad_emb, grad_W_hid, grad_b_hid, grad_W_out, grad_b_out]
            else:
                logits = batch_in @ W_out + b_out
                g_out = softmax_grad(logits, yy)
                grad_W_out = batch_in.T @ g_out + L2 * W_out
                grad_b_out = g_out.sum(0)
                grad_emb = np.zeros_like(emb)
                for k in range(n_cat):
                    for b_idx, row_idx in enumerate(idx):
                        grad_emb[k, xb[b_idx, cat_idx[k]]] += g_out[b_idx, k*E:(k+1)*E]
                grad_emb += L2 * emb
                grads = [grad_emb, np.zeros_like(W_hid), np.zeros_like(b_hid), grad_W_out, grad_b_out]

            t = 1  # simple Adam step counter per batch (approximate)
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x454D424C  # "L B M E"
    type_arr = bytearray(1 if t == "categorical" else 0 for t in types)
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<5i", magic, d, E, H, C))
        f.write(type_arr)
        # embeddings
        for k in range(n_cat):
            f.write(emb[k].astype("<f4").tobytes())
        # ordinal stats
        for k in range(n_ord):
            f.write(struct.pack("<2f", ord_mean[k], ord_std[k]))
        # output layer
        f.write(W_out.astype("<f4").tobytes())
        f.write(b_out.astype("<f4").tobytes())
        if H:
            f.write(W_hid.astype("<f4").tobytes())
            f.write(b_hid.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    unsigned char *type = wb + 20;
    int n_cat = 0, n_ord = 0;
    for (int j = 0; j < d; j++) {
        if (type[j]) n_cat++; else n_ord++;
    }
    int input_dim = n_cat * E + n_ord;
    int W = H ? H : input_dim;

    const float *p = (const float *)(type + d);
    const float *emb = p;
    p += n_cat * 256 * E;
    const float *ord_mean = p;
    p += n_ord;
    const float *ord_std = p;
    p += n_ord;
    const float *W_out = p;
    p += W * C;
    const float *b_out = p;
    p += C;
    const float *W_hid = H ? p : NULL;
    const float *b_hid = H ? (p + input_dim * H) : NULL;

    long expected = 20 + d + 4L * (n_cat * 256 * E + 2 * n_ord + W * C + C + (H ? input_dim * H + H : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_vec = malloc(sizeof(float) * input_dim);
    float *hid = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int cat_ptr = 0, ord_ptr = 0;
        for (int j = 0; j < d; j++) {
            if (type[j]) {
                const float *e = emb + (cat_ptr * 256 + xi[j]) * E;
                memcpy(in_vec + cat_ptr * E, e, E * sizeof(float));
                cat_ptr++;
            } else {
                in_vec[n_cat * E + ord_ptr] = (xi[j] - ord_mean[ord_ptr]) / ord_std[ord_ptr];
                ord_ptr++;
            }
        }

        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b_hid[h];
                const float *w = W_hid + (long)input_dim * h;
                for (int v = 0; v < input_dim; v++) sum += in_vec[v] * w[v];
                hid[h] = sum > 0 ? sum : 0;
            }
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                const float *w = W_out + (long)H * c;
                for (int h = 0; h < H; h++) sum += hid[h] * w[h];
                scores[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                const float *w = W_out + (long)input_dim * c;
                for (int v = 0; v < input_dim; v++) sum += in_vec[v] * w[v];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```