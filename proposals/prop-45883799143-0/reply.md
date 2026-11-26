```hypothesis
Family: label-emb-mlp (embedding + one hidden layer MLP). 
Why: Learns dense embeddings for each of the 14 features (8-d), capturing non-linear interactions better than linear one-hot, while keeping inference fast (112-d input, 16 hidden units). 
Failure: If dev accuracy <= 0.8524 (linear) or Ir/item >= 2325 (mlp16), the model does not improve the accuracy-cost trade-off.
```

```python
import json, math, struct, sys
import numpy as np

E = 8
H = 16
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

    # embeddings: (d, 256, E)
    limit = math.sqrt(6.0 / (256 + E))
    emb = rng.uniform(-limit, limit, (d, 256, E)).astype(np.float32)

    input_dim = d * E
    limit1 = math.sqrt(6.0 / (input_dim + H))
    W1 = rng.uniform(-limit1, limit1, (input_dim, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    limit2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-limit2, limit2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(idx)

            # concat embeddings
            concat = np.zeros((bs, input_dim), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j][xb[:, j]]

            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            # softmax grad
            lmax = logits.max(axis=1, keepdims=True)
            exp = np.exp(logits - lmax)
            probs = exp / exp.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs

            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = concat.T @ grad_hidden_pre
            grad_b1 = grad_hidden_pre.sum(0)
            grad_concat = grad_hidden_pre @ W1.T

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_concat[:, j*E:(j+1)*E])

            grad_W2 += L2 * W2
            grad_W1 += L2 * W1
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                mhat = m[i] / (1 - beta1 ** t)
                vhat = v[i] / (1 - beta2 ** t)
                p -= LR * mhat / (np.sqrt(vhat) + eps)

    magic = 0x4C454D31
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x4C454D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 14 || E != 8 || H != 16 || C != 2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20 
        + (long)d * 256 * E * sizeof(float)
        + (long)d * E * H * sizeof(float)
        + (long)H * sizeof(float)
        + (long)H * C * sizeof(float)
        + (long)C * sizeof(float);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) concat[j * E + e_idx] = e[e_idx];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```