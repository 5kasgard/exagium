```hypothesis
Family: label-shared-embedding. A single 256×E embedding matrix is shared across all 16 context positions, followed by a ReLU hidden layer of size H and a linear output layer. Sharing embeddings reduces parameters and captures character similarities across positions, improving generalization over per-position one-hot (label-linear). The model is trained with Adam and L2 regularization. If dev accuracy does not exceed label-linear (0.3375) or inference cost per item is significantly higher, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
E = 16          # embedding dimension
H = 64          # hidden layer size
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
    d = X.shape[1]  # should be 16
    C = card["classes"]  # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward
            emb_x = emb[xb]                 # (B, d, E)
            flat = emb_x.reshape(B, d * E)  # (B, d*E)
            hidden_pre = flat @ W1 + b1     # (B, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2       # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = flat.T @ grad_hidden_pre
            grad_b1 = grad_hidden_pre.sum(axis=0)
            grad_flat = grad_hidden_pre @ W1.T
            grad_emb_x = grad_flat.reshape(B, d, E)
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, xb, grad_emb_x)

            # L2 regularization
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x5348454D  # "SHEM"
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x5348454D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20
        + 4L * 256 * E
        + 4L * (d * E) * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e_idx = 0; e_idx < E; e_idx++) {
                    sum += e[e_idx] * W1[((long)j * E + e_idx) * H + h];
                }
                hidden[h] += sum;
            }
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hval = hidden[h];
            if (hval == 0.0f) continue;
            for (int c = 0; c < C; c++) {
                logits[c] += hval * W2[(long)h * C + c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```