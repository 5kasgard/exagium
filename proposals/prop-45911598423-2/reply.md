```hypothesis
Propose "label-embedding": a shallow embedding model that learns a dense vector for each (position, byte) pair, sums the 16 vectors, and applies a linear classifier. This captures pairwise interactions through the shared embedding space and should outperform the linear one-hot model (label-linear) at similar or lower inference cost because the embedding dimension is small (e.g., 32). If it does not beat label-linear on dev accuracy, or if its inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
"""label-embedding: shallow embedding model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D4244, int32 d, C, d_in, vocab; float32 emb[d_in][vocab][d], W_out[d][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all)
SEED = 1
EMB_DIM = 32
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 30
MAGIC = 0x454D4244  # "EMBD" little-endian


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d_in = card["input"]["bytes"]
    C = card["classes"]
    vocab = 256

    X_recs = read_records(sys.argv[2])
    y_recs = read_records(sys.argv[3])
    n = len(X_recs)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(n, d_in)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    # Split: last 20% for validation (tuning), but we fix hyperparams and train on all
    rng = np.random.default_rng(SEED)

    # Initialize embeddings and output layer
    emb = (rng.normal(0, 0.01, (d_in, vocab, EMB_DIM))).astype(np.float32)
    W_out = (rng.normal(0, 0.01, (EMB_DIM, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def softmax_cross_entropy_grad(logits, targets):
        # logits: (batch, C), targets: (batch,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return probs / len(targets)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, d_in)
            yb = y[idx]          # (batch,)

            # Forward: sum embeddings
            # emb_sum[b, k] = sum_j emb[j, xb[b, j], k]
            emb_sum = np.zeros((len(yb), EMB_DIM), dtype=np.float32)
            for j in range(d_in):
                emb_sum += emb[j, xb[:, j]]

            logits = emb_sum @ W_out + b_out
            grad_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            # Gradients
            grad_W_out = emb_sum.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_emb_sum = grad_logits @ W_out.T

            # Backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d_in):
                # For each position j, accumulate grad_emb_sum into grad_emb[j, xb[:, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W_out, grad_W_out, m_W, v_W),
                (b_out, grad_b_out, m_b, v_b),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<5i", MAGIC, EMB_DIM, C, d_in, vocab))
        f.write(emb.astype("<f4").tobytes())
        f.write(W_out.astype("<f4").tobytes())
        f.write(b_out.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], d_in = hd[3], vocab = hd[4];
    if (d_in != 16 || vocab != 256 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 20L + 4L * ((long)d_in * vocab * d + (long)d * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *emb = (float *)(wb + 20);
    float *W_out = emb + (long)d_in * vocab * d;
    float *b_out = W_out + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * d);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_in) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_in); return 3; }

        for (int kk = 0; kk < d; kk++) emb_sum[kk] = 0.0f;
        for (int j = 0; j < d_in; j++) {
            const float *e = emb + ((long)j * vocab + xi[j]) * d;
            for (int kk = 0; kk < d; kk++) emb_sum[kk] += e[kk];
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int kk = 0; kk < d; kk++) sum += emb_sum[kk] * W_out[(long)kk * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```