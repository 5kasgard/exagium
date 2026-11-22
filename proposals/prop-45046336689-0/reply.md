```hypothesis
Family: label-emb (embedding-based logistic regression). Each of the 14 byte columns receives a learned embedding vector of dimension D. The record representation is the sum of its column embeddings, passed through a single linear layer. This captures cross-column similarities through the shared embedding space, improving over one-hot linear models while keeping inference extremely fast (14×D additions + D multiplications). We expect it to beat label-linear (0.8524) on dev accuracy because embeddings can learn that, e.g., similar occupation codes behave similarly. If dev accuracy ≤ 0.855, the added parameters do not justify the complexity.
```

```python
import json, struct, sys
import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
D = 16
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "Only binary classification supported"
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: (d, 256, D)
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W_out = rng.normal(0, 0.01, size=(D,)).astype(np.float32)
    bias_out = np.float32(0.0)

    # Adam state
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W_out)
    v_W = np.zeros_like(W_out)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            # Forward: sum embeddings per record
            emb_sum = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]

            logits = emb_sum @ W_out + bias_out
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = probs - yb.astype(np.float32)  # (B,)

            # Gradients
            grad_W_out = (emb_sum.T @ grad_logits) / B + L2 * W_out
            grad_bias_out = grad_logits.mean()
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_logits[:, None])
            grad_emb /= B
            grad_emb += L2 * emb

            # Adam update
            t += 1
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            emb -= LR * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)

            m_W = beta1 * m_W + (1 - beta1) * grad_W_out
            v_W = beta2 * v_W + (1 - beta2) * (grad_W_out * grad_W_out)
            W_out -= LR * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_bias_out
            v_b = beta2 * v_b + (1 - beta2) * (grad_bias_out * grad_bias_out)
            bias_out -= LR * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)

    # Save weights: magic, d, D, C; embeddings[d][256][D]; W_out[D]; bias_out
    magic = 0x454D424C  # "LBME"
    out = struct.pack("<4I", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += struct.pack("<f", bias_out)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C != 2) { fprintf(stderr, "only binary classification supported\n"); return 3; }
    long expected = 16L + (long)d * 256 * D * 4 + D * 4 + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * D;
    const float *bias_out = W_out + D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum = (float *)malloc(sizeof(float) * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) sum[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) sum[v] += e[v];
        }
        float logit = *bias_out;
        for (int v = 0; v < D; v++) logit += sum[v] * W_out[v];
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```