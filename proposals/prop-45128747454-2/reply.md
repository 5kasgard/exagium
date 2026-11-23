```hypothesis
Family: label-ngram (linear with unigram + hashed bigram features).
Why: Character-level language modeling relies heavily on local n-grams (e.g., "th"→"e").
Label-linear uses only positional unigrams; adding hashed bigrams (15 adjacent pairs → 8192 buckets)
captures pairwise interactions with minimal parameter overhead (~3.2 MB weights).
Inference stays fast: 31 table lookups per item (16 unigram + 15 bigram) vs 16 for linear.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or Ir/item > 10000 (2× linear cost).
```

```python
import json, math, struct, sys
import numpy as np

HASH_SIZE = 8192
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    scale_U = math.sqrt(6.0 / (d * 256))
    scale_B = math.sqrt(6.0 / HASH_SIZE)
    U = (rng.uniform(-1, 1, (d, 256, C)) * scale_U).astype(np.float32)
    B = (rng.uniform(-1, 1, (HASH_SIZE, C)) * scale_B).astype(np.float32)
    bias = np.zeros(C, np.float32)

    m_U, v_U = np.zeros_like(U), np.zeros_like(U)
    m_B, v_B = np.zeros_like(B), np.zeros_like(B)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)

            scores = np.tile(bias, (bsz, 1))
            for j in range(d):
                scores += U[j, xb[:, j]]
            for j in range(d - 1):
                h = ((j << 16) | (xb[:, j].astype(np.int32) << 8) | xb[:, j+1].astype(np.int32)) & (HASH_SIZE - 1)
                scores += B[h]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad = probs / bsz

            grad_bias = grad.sum(axis=0)
            grad_U = np.zeros_like(U)
            grad_B = np.zeros_like(B)

            for j in range(d):
                for c in range(C):
                    np.add.at(grad_U[j, :, c], xb[:, j], grad[:, c])

            for j in range(d - 1):
                h = ((j << 16) | (xb[:, j].astype(np.int32) << 8) | xb[:, j+1].astype(np.int32)) & (HASH_SIZE - 1)
                for c in range(C):
                    np.add.at(grad_B[:, c], h, grad[:, c])

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_b = beta1 * m_b + (1 - beta1) * grad_bias
            v_b = beta2 * v_b + (1 - beta2) * grad_bias * grad_bias
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

            grad_U += L2 * U
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U * grad_U
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U -= LR * m_U_hat / (np.sqrt(v_U_hat) + eps)

            grad_B += L2 * B
            m_B = beta1 * m_B + (1 - beta1) * grad_B
            v_B = beta2 * v_B + (1 - beta2) * grad_B * grad_B
            m_B_hat = m_B / (1 - beta1 ** t)
            v_B_hat = v_B / (1 - beta2 ** t)
            B -= LR * m_B_hat / (np.sqrt(v_B_hat) + eps)

    magic = 0x4D52474E
    out = struct.pack("<5i", magic, d, C, HASH_SIZE, 0)
    out += U.astype("<f4").tobytes()
    out += B.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4D52474E) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], hash_size = hd[3], H = hd[4];
    if (H != 0) { fprintf(stderr, "only linear supported\n"); return 3; }
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad params\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * C + (long)hash_size * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *U = (const float *)(hd + 5);
    const float *B = U + (long)d * 256 * C;
    const float *bias = B + (long)hash_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, sizeof(float) * C);

        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += u[c];
        }

        for (int j = 0; j < d - 1; j++) {
            uint32_t h = ((j << 16) | (xi[j] << 8) | xi[j+1]) & (hash_size - 1);
            const float *b = B + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += b[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```