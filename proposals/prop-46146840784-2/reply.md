```hypothesis
Family: label-emb (low-dimensional embedding + linear)
Why: 16 categorical bytes (256 values each) – a linear model on one-hot (label-linear) uses 16*256*C parameters.
Embedding each (position, byte) into D=64 dimensions and then a linear layer to 65 classes reduces parameters to
16*256*64 + 64*65 ≈ 266k vs 16*256*65 ≈ 266k (similar size) but adds non-linear representation power via the
embedding lookup. Inference sums 16 vectors of length 64 then does a 64x65 matmul – cheaper than label-linear's
16*256*65 adds. Expected to match or beat label-linear accuracy (0.3375) at lower CPU cost.
Failure: dev accuracy < 0.3375 or instructions/item > label-linear's 7360.
```

```python
"""label-emb: embedding + linear classifier for fixed-size byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic uint32 = 0x4C424D45 ('EMBL')
  d, D, C int32
  embeddings[d][256][D] float32
  W_out[D][C] float32
  b_out[C] float32
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen a priori, no tuning split used)
D = 64                # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: (d, 256, D)
    E = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # Output layer
    W_out = (rng.normal(0, 0.01, (D, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W_out), np.zeros_like(W_out)
    mb, vb = np.zeros_like(b_out), np.zeros_like(b_out)
    t = 0

    def softmax_grad(logits, targets):
        # logits: (B, C), targets: (B,)
        max_logit = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return probs / len(targets)   # (B, C)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]      # (B, d)
            yb = y[idx]      # (B,)

            # Forward: sum embeddings -> (B, D)
            h = np.zeros((len(yb), D), np.float32)
            for j in range(d):
                h += E[j, xb[:, j]]   # broadcast add

            logits = h @ W_out + b_out            # (B, C)
            grad_logits = softmax_grad(logits, yb) # (B, C)

            # Gradients
            grad_W_out = h.T @ grad_logits + L2 * W_out          # (D, C)
            grad_b_out = grad_logits.sum(axis=0)                 # (C,)
            grad_h = grad_logits @ W_out.T                       # (B, D)

            grad_E = np.zeros_like(E)
            for j in range(d):
                # accumulate gradient for each byte value
                np.add.at(grad_E[j], xb[:, j], grad_h)
            grad_E += L2 * E

            t += 1
            # Adam updates
            for param, grad, m, v in [
                (E, grad_E, mE, vE),
                (W_out, grad_W_out, mW, vW),
                (b_out, grad_b_out, mb, vb)
            ]:
                m *= ADAM_B1; m += (1 - ADAM_B1) * grad
                v *= ADAM_B2; v += (1 - ADAM_B2) * grad * grad
                m_hat = m / (1 - ADAM_B1 ** t)
                v_hat = v / (1 - ADAM_B2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    # Pack weights
    magic = 0x4C424D45  # 'EMBL'
    header = struct.pack("<4i", magic, d, D, C)
    emb_bytes = E.astype("<f4").tobytes()
    w_bytes = W_out.astype("<f4").tobytes()
    b_bytes = b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + w_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   uint32 magic = 0x4C424D45
 *   int32 d, D, C
 *   float32 embeddings[d][256][D]
 *   float32 W_out[D][C]
 *   float32 b_out[C]
 */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W_out = E + (long)d * 256 * D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * D);
    float *logits = (float *)malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld len %u != %d\n", i, k, d); return 3; }

        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *emb = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += emb[u];
        }

        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int u = 0; u < D; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W_out + (long)u * C;
            for (int c = 0; c < C; c++) logits[c] += hu * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```