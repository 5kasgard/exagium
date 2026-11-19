```hypothesis
Family: label-emb (embedding + linear). Each of the 16 context bytes gets a learned D-dimensional embedding; the 16 embeddings are summed and fed to a linear output layer (65 classes). This factorizes the full 16*256*65 weight matrix into 16*256*D + D*65 parameters, acting as a regularized low-rank approximation that should generalize better than the full linear model on this small dataset. With D=32 the model has ~133k parameters (vs 266k for linear) and similar inference cost (≈2600 FLOPs/item). If dev accuracy does not exceed 0.34, the factorization is too restrictive for this task.
```

```python
"""label-emb: embedding sum + linear output for 16-byte context next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports: magic 0x454D4231, d=16, D, C=65; float32 embeddings[d][256][D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 32               # embedding dimension
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
    d, C = X.shape[1], card["classes"]          # d=16, C=65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Xavier init for embeddings: uniform(-sqrt(6/fan_in), sqrt(6/fan_in)), fan_in=256
    lim_emb = math.sqrt(6.0 / 256)
    emb = rng.uniform(-lim_emb, lim_emb, (d, 256, D)).astype(np.float32)

    # Xavier init for output layer: fan_in=D, fan_out=C
    lim_out = math.sqrt(6.0 / (D + C))
    W = rng.uniform(-lim_out, lim_out, (D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [emb, W, b]
    # Adam state: m, v for each parameter array
    state = [(np.zeros_like(p, dtype=np.float32), np.zeros_like(p, dtype=np.float32)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, 16)
            yb = y[idx]                 # (B,)
            B = len(yb)

            # Forward: sum embeddings
            h = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]   # (B, D)

            logits = h @ W + b          # (B, C)
            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Gradients
            dW = h.T @ dlogits + L2 * W
            db = dlogits.sum(axis=0)
            dh = dlogits @ W.T           # (B, D)

            # Embedding gradients: accumulate per position
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dh)
            demb += L2 * emb

            grads = [demb, dW, db]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, D, C; then embeddings (d*256*D), W (D*C), b (C)
    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<4i", magic, d, D, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x454D4231), d, D, C; float32 emb[d][256][D], W[D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = sum_j emb[j][xi[j]]
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *row = emb + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) h[v] += row[v];
        }

        // scores = h * W + b
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int v = 0; v < D; v++) sum += h[v] * W[(long)v * C + c];
            scores[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```