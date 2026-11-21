```hypothesis
Family: label-embed (embedding-based 2-layer MLP). Each of the 16 context positions gets a learned 8-dim embedding per byte value; concatenated embeddings (128-d) feed a 64-unit ReLU layer then 65-way output. Beats linear by sharing stats across characters (e.g., vowels cluster) and modeling position interactions non-linearly. Fails if 8-dim embeddings are too small to capture character relationships or if 25 epochs underfits.
```
```python
import json, math, struct, sys

import numpy as np

E = 8
H = 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, E]
    Emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Layer 1: [H, d*E]
    k = d * E
    W1 = (rng.normal(0, math.sqrt(2.0 / k), (H, k))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Layer 2: [C, H]
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (C, H))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            emb = Emb[np.arange(d)[None, :], xb]  # [bs, d, E]
            flat = emb.reshape(bs, -1)            # [bs, d*E]
            h = flat @ W1.T + b1                  # [bs, H]
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2.T + b2           # [bs, C]

            # Softmax gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Backprop
            grad_W2 = grad_logits.T @ h_relu + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            grad_h_relu = grad_logits @ W2
            grad_h = grad_h_relu * (h > 0)

            grad_W1 = grad_h.T @ flat + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            grad_flat = grad_h @ W1
            grad_emb = grad_flat.reshape(bs, d, E)
            grad_Emb = np.zeros_like(Emb)
            for j in range(d):
                np.add.at(grad_Emb[j], xb[:, j], grad_emb[:, j])
            grad_Emb += L2 * Emb

            grads = [grad_Emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: magic, d, E, H, C; then Emb, W1, b1, W2, b2 (all float32 LE)
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += Emb.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)H * d * E + H + (long)C * H + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)H * d * E;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *src = Emb + ((long)j * 256 + xi[j]) * E;
            float *dst = flat + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            const float *w = W1 + (long)u * d * E;
            for (int k = 0; k < d * E; k++) sum += flat[k] * w[k];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            const float *w = W2 + (long)c * H;
            for (int u = 0; u < H; u++) sum += h[u] * w[u];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```