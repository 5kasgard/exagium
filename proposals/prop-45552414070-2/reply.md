```hypothesis
Family: label-emb (position-specific character embeddings + linear). Each of the 16 context positions gets a 256×E embedding table (E=16). Concatenated embeddings (256-dim) feed a linear layer to 65 classes. Trained with Adam (lr=3e-3, L2=1e-4) for 25 epochs. Embeddings capture character similarities (e.g., vowels cluster) and position-specific patterns, reducing parameters vs one-hot linear (4096→256 features) and improving generalization. If dev accuracy ≤ 0.3375, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

E = 16
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    W = rng.normal(0, 0.01, size=(C, d * E)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            emb_concat = np.zeros((B, d * E), dtype=np.float32)
            for j in range(d):
                emb_concat[:, j * E:(j + 1) * E] = emb[j, x_batch[:, j], :]

            scores = emb_concat @ W.T + b

            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_scores = probs / B

            grad_W = grad_scores.T @ emb_concat + L2 * W
            grad_b = np.sum(grad_scores, axis=0)

            grad_emb_concat = grad_scores @ W
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                g = grad_emb_concat[:, j * E:(j + 1) * E]
                np.add.at(grad_emb[j], x_batch[:, j], g)
            grad_emb += L2 * emb

            t += 1
            m_emb = 0.9 * m_emb + 0.1 * grad_emb
            v_emb = 0.999 * v_emb + 0.001 * grad_emb * grad_emb
            m_emb_hat = m_emb / (1 - 0.9 ** t)
            v_emb_hat = v_emb / (1 - 0.999 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + 1e-8)

            m_W = 0.9 * m_W + 0.1 * grad_W
            v_W = 0.999 * v_W + 0.001 * grad_W * grad_W
            m_W_hat = m_W / (1 - 0.9 ** t)
            v_W_hat = v_W / (1 - 0.999 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            m_b_hat = m_b / (1 - 0.9 ** t)
            v_b_hat = v_b / (1 - 0.999 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

    magic = 0x44424D45
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)C * d * E + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)C * d * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *w = W + (long)c * d * E;
            for (int f = 0; f < d * E; f++) sum += w[f] * concat[f];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```