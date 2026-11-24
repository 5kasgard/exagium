```hypothesis
Family: label-emb (position-specific embeddings + linear). Each of the 16 context positions gets a 256×E embedding table; the concatenated 16×E vector feeds a linear layer to 65 classes. Embeddings capture character similarities (e.g., vowels cluster) and reduce parameters vs. one-hot linear (16×256→16×E). With E=8 this has ~41k parameters, trains fast, and should exceed label-linear’s 0.3375 accuracy. Failure to beat label-linear would indicate embeddings don’t help for this fixed-window character prediction.
```

```python
"""label-emb: position-specific embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

E = 8                 # embedding size per position
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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: embeddings[d][256][E], W[d*E][C], b[C]
    embeddings = [(rng.standard_normal((256, E), dtype=np.float32) * math.sqrt(2.0 / 256)) for _ in range(d)]
    W = (rng.standard_normal((d * E, C), dtype=np.float32) * math.sqrt(2.0 / (d * E)))
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb = [np.zeros_like(e) for e in embeddings]
    v_emb = [np.zeros_like(e) for e in embeddings]
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: build concatenated embeddings
            x_emb = np.empty((bs, d * E), dtype=np.float32)
            for p in range(d):
                x_emb[:, p*E:(p+1)*E] = embeddings[p][xb[:, p]]

            logits = x_emb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs

            # Gradients
            grad_W = x_emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_x_emb = grad_logits @ W.T

            # Embedding gradients via one-hot accumulation
            grad_emb = [np.zeros_like(e) for e in embeddings]
            for p in range(d):
                slice_ = grad_x_emb[:, p*E:(p+1)*E]
                idx_p = xb[:, p]
                # one-hot scatter add
                for i in range(bs):
                    grad_emb[p][idx_p[i]] += slice_[i]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)
            # embeddings
            for p in range(d):
                m_emb[p] = beta1 * m_emb[p] + (1 - beta1) * grad_emb[p]
                v_emb[p] = beta2 * v_emb[p] + (1 - beta2) * grad_emb[p] * grad_emb[p]
                m_hat = m_emb[p] / (1 - beta1 ** t)
                v_hat = v_emb[p] / (1 - beta2 ** t)
                embeddings[p] -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights: magic, d, E, C, embeddings (d*256*E), W (d*E*C), b (C)
    magic = 0x4C424D45  # 'EMBL' little-endian
    out = struct.pack("<4i", magic, d, E, C)
    for p in range(d):
        out += embeddings[p].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index. */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * E;
            memcpy(x_emb + p * E, e, E * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d * E; j++) {
            float xv = x_emb[j];
            if (xv == 0.0f) continue;
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += xv * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```