```hypothesis
Family: label-emb (per-feature embeddings + linear). Each of the 14 input bytes indexes a 4-d embedding vector; concatenated embeddings (56 dims) feed a linear 2-class head. This captures non-linear per-feature effects and cross-feature interactions via the linear layer, while inference needs only 14 small lookups and 56×2 multiply-adds. Expect >0.8524 dev accuracy (beating label-linear) at ~200 Ir/item. If accuracy ≤0.8524, the task is essentially linear in the one-hot encoding.
```
```python
"""label-emb: per-feature embeddings (dim=4) + linear head.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D4244, d, E, C; float32 embeddings[d][256][E], W_out[d*E][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
EMBED_DIM = 4
LR = 0.01
L2 = 1e-5
BATCH = 64
EPOCHS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    E = EMBED_DIM

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: list of d arrays (256, E)
    embeddings = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    W_out = rng.normal(0, 0.01, (d * E, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = embeddings + [W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward: concatenate embeddings
            emb_concat = np.empty((bs, d * E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]

            logits = emb_concat @ W_out + b_out
            probs = softmax(logits)
            probs[np.arange(bs), yb] -= 1.0
            dlogits = probs / bs

            # gradients
            dW_out = emb_concat.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(0)
            demb_concat = dlogits @ W_out.T  # (bs, d*E)

            grads = []
            for j in range(d):
                dj = demb_concat[:, j*E:(j+1)*E]
                g = np.zeros((256, E), np.float32)
                np.add.at(g, xb[:, j], dj)
                g += L2 * embeddings[j]
                grads.append(g)
            grads.append(dW_out)
            grads.append(db_out)

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= beta1; m[i] += (1 - beta1) * gr
                v[i] *= beta2; v[i] += (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    out = struct.pack("<4i", 0x454D4244, d, E, C)
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights: magic 0x454D4244, int32 d, E, C; float32 embeddings[d][256][E], W_out[d*E][C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 14 || E != 4 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = concat + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }

        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int c = 0; c < C; c++) {
            const float *w = W_out + c;
            float sum = 0.0f;
            for (int m = 0; m < d * E; m++) sum += concat[m] * w[m * C];
            logits[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```