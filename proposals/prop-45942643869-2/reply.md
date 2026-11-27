```hypothesis
Family: label-embsum (positional embedding sum + linear). Each of the 16 context positions has a 256×E embedding table; the 16 vectors are summed into a single E-vector, then a linear layer predicts 65 classes. This shares statistical strength across positions while keeping inference extremely cheap (16×E adds + E×65 MACs). With E=32 it should beat label-linear's 0.3375 accuracy at similar instruction count. Failure would be dev accuracy ≤ 0.34 or Ir/item > 10k.
```

```python
"""label-embsum: positional embedding sum + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, int32 d,E,C; float32 emb[d][256][E], W[E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 20


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
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # embeddings: [d, 256, E]
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / (E + C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)

    n = X.shape[0]
    steps = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward: sum embeddings
            # emb_sum[bs, E]
            emb_sum = np.zeros((bs, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]  # (bs, E)
            logits = emb_sum @ W + b  # (bs, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient on logits
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs

            # gradients
            grad_W = emb_sum.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb_sum = grad_logits @ W.T  # (bs, E)

            # backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # grad_emb[j, byte, :] += sum over batch where xb[:,j]==byte of grad_emb_sum
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            steps += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # update embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_emb_hat = m_emb / (1 - beta1 ** steps)
            v_emb_hat = v_emb / (1 - beta2 ** steps)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

            # update W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_W_hat = m_W / (1 - beta1 ** steps)
            v_W_hat = v_W / (1 - beta2 ** steps)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)

            # update b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_b_hat = m_b / (1 - beta1 ** steps)
            v_b_hat = v_b / (1 - beta2 ** steps)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # write weights
    magic = 0x454D4244
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embsum inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += eptr[e];
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int e = 0; e < E; e++) sum += acc[e] * W[(long)e * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```