```hypothesis
Family: label-emblin (embedding + linear). Learns a 32-d embedding per byte at each of the 16 positions, then a linear layer on the concatenated 512-d vector. This low-rank factorization of the one-hot linear model (4096*65 -> 16*256*32 + 512*65 ≈ 164k params) should generalize better on 24k samples. If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds label-linear, the hypothesis fails.
```

```python
"""label-emblin: embedding per position + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# hyperparameters
D = 32                     # embedding dimension
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20            # will be capped by SAMPLES/n

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
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # parameters: embeddings (d, 256, D), W (d*D, C), b (C)
    emb = (rng.normal(0, 1, (d, 256, D)) * math.sqrt(2.0 / (256 + D))).astype(np.float32)
    W = (rng.normal(0, 1, (d * D, C)) * math.sqrt(2.0 / (d * D + C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = xb.shape[0]

            # gather embeddings: (B, d, D)
            emb_batch = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, D)
            flat = emb_batch.reshape(B, d * D)                      # (B, d*D)

            # logits
            logits = flat @ W + b                                   # (B, C)
            # softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (B, C)

            # gradient wrt logits
            probs[np.arange(B), yb] -= 1.0
            probs /= B                                               # (B, C)

            # gradients
            grad_W = flat.T @ probs + L2 * W                         # (d*D, C)
            grad_b = probs.sum(axis=0)                               # (C,)
            grad_flat = probs @ W.T                                  # (B, d*D)
            grad_emb_batch = grad_flat.reshape(B, d, D)              # (B, d, D)

            # accumulate grad_emb for each (position, byte)
            grad_emb = np.zeros_like(emb)
            # vectorized scatter-add
            np.add.at(grad_emb, (np.arange(d), xb.T), grad_emb_batch.transpose(1, 0, 2))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    # magic: 0x4E494C42 ("BLIN" little-endian), d, D, C
    magic = 0x4E494C42
    header = struct.pack("<4i", magic, d, D, C)
    # embeddings: (d, 256, D) -> flat float32
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-emblin infer: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x4E494C42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = (float *)malloc(sizeof(float) * d * D);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            memcpy(flat + p * D, e, D * sizeof(float));
        }

        // linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d * D; j++) {
            float f = flat[j];
            if (f == 0.0f) continue;
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += f * w[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(flat); free(scores);
    return 0;
}
```