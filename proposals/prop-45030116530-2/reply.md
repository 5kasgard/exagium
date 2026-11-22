```hypothesis
Family: label-embed (positional character embeddings + linear output).  
Why: The task predicts the next character from 16 context bytes; each byte is one of 65 classes. A linear model on one-hot per position (label-linear) uses ~266k parameters but cannot share statistical strength across characters. Learned embeddings (dimension 32) allow similar characters to have similar representations, reducing parameters to ~133k and acting as a regularizer. The model is a single embedding lookup per position followed by a sum and a small dense layer, which is extremely fast at inference.  
Failure signal: If dev accuracy does not exceed 0.34 (label-linear's 0.3375) or inference cost exceeds label-linear, the hypothesis is wrong.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 32               # embedding dimension
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 256
EPOCHS = 20
SAMPLES = 600000     # max training samples

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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: (d, 256, D)
    E = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    W_out = rng.normal(0, 0.01, (D, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    # Adam state
    m_E, v_E = np.zeros_like(E), np.zeros_like(E)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    max_epochs = min(EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward: gather embeddings and sum over positions
            # emb shape (B, d, D)
            emb = E[np.arange(d)[None, :, None], x_batch[:, :, None], np.arange(D)[None, None, :]]
            emb_sum = emb.sum(axis=1)                     # (B, D)
            logits = emb_sum @ W_out + b_out              # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            probs /= B

            # Gradients
            grad_W = emb_sum.T @ probs + L2 * W_out       # (D, C)
            grad_b = probs.sum(axis=0) + L2 * b_out       # (C,)
            grad_emb_sum = probs @ W_out.T                # (B, D)

            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], x_batch[:, j], grad_emb_sum)
            grad_E += L2 * E

            # Adam update
            t += 1
            for param, grad, m, v in [(E, grad_E, m_E, v_E),
                                      (W_out, grad_W, m_W, v_W),
                                      (b_out, grad_b, m_b, v_b)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<4i", magic, d, D, C)
    out = header + E.astype("<f4").tobytes() + W_out.astype("<f4").tobytes() + b_out.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 1024) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W_out = E + (long)d * 256 * D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) emb_sum[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) emb_sum[kk] += e[kk];
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int kk = 0; kk < D; kk++) sum += emb_sum[kk] * W_out[(long)kk * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```