```hypothesis
Family: label-embsum (position-specific embeddings summed, ReLU, linear). 
Why: The task is next-character prediction from 16 bytes. A linear model (label-linear) treats each position independently with a 256×65 weight matrix per position. Our model learns a dense embedding per byte per position (16×256×D), sums them, applies a ReLU non-linearity, then a linear layer to 65 classes. This adds cross-position interactions via the shared hidden layer and non-linearity, which should improve accuracy over pure linear at similar parameter count (D=128 gives ~530k params vs linear's 266k). Inference is fast: 16 embedding lookups, a D-vector sum, ReLU, and a D×65 matvec.
Failure mode: If accuracy does not exceed label-linear (0.3375), the bottleneck may be the sum pooling losing positional order, or the model may overfit the small training set (24k items) without stronger regularization.
```

```python
import json, struct, sys
import numpy as np

# Hyperparameters (chosen after tuning on last 20% of train.in)
D = 128
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: embeddings [d, 256, D], bias1 [D], W_out [D, C], b_out [C]
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    bias1 = np.zeros(D, dtype=np.float32)
    W_out = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_b1 = np.zeros_like(bias1); v_b1 = np.zeros_like(bias1)
    m_W = np.zeros_like(W_out); v_W = np.zeros_like(W_out)
    m_b2 = np.zeros_like(b_out); v_b2 = np.zeros_like(b_out)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward: sum embeddings + bias1 -> ReLU -> logits
            hidden = np.zeros((B, D), dtype=np.float32)
            for p in range(d):
                hidden += emb[p, x_batch[:, p]]
            hidden += bias1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W_out + b_out

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = probs / B

            # Backprop
            grad_hidden_relu = grad_logits @ W_out.T
            grad_hidden = grad_hidden_relu * (hidden > 0)

            grad_W_out = hidden_relu.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)

            grad_bias1 = grad_hidden.sum(axis=0)
            grad_emb = np.zeros_like(emb)
            for p in range(d):
                np.add.at(grad_emb[p], x_batch[:, p], grad_hidden)
            grad_emb += L2 * emb

            t += 1
            def adam_step(param, grad, m, v):
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

            adam_step(emb, grad_emb, m_emb, v_emb)
            adam_step(bias1, grad_bias1, m_b1, v_b1)
            adam_step(W_out, grad_W_out, m_W, v_W)
            adam_step(b_out, grad_b_out, m_b2, v_b2)

    # Write weights.bin: magic, d, D, C, then embeddings, bias1, W_out, b_out (all float32 LE)
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += bias1.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
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
    if (d != 16 || C != 65 || D <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *bias1 = emb + (long)d * 256 * D;
    const float *W_out = bias1 + D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * D);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < D; j++) hidden[j] = bias1[j];
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) hidden[j] += e[j];
        }
        for (int j = 0; j < D; j++) if (hidden[j] < 0.0f) hidden[j] = 0.0f;

        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < D; j++) sum += hidden[j] * W_out[(long)j * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```