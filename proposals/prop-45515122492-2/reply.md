```hypothesis
Family: label-lr (Low-Rank Linear). Factorizes each position's 256×65 weight matrix into a 256×D embedding and a shared D×65 output layer (D=32). For character prediction, similar bytes (e.g., vowels, punctuation) exhibit similar transition patterns; the low-rank structure forces this sharing, acting as strong regularization vs label-linear's 266K free parameters. With ~82K params it should match or beat 0.3375 dev accuracy at comparable CPU cost (16·D adds + D·65 FMAs per item vs 16·65 adds). Failure: dev accuracy < 0.33 or Ir/item > 8000.
```
```python
"""label-lr: Low-rank linear model for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports: magic 0x4C523130, d, D, C; float32 E[d*256*D], W[D*C]."""
import json, math, struct, sys

import numpy as np

D = 32
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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: (d, 256, D)
    E = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # Output weights: (D, C)
    W = (rng.normal(0, 0.01, (D, C))).astype(np.float32)

    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    t = 0

    def softmax_grad(logits, y_idx):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs.astype(np.float32) / len(y_idx)

    steps_per_epoch = math.ceil(n / BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    for step in range(max_steps):
        idx = rng.choice(n, BATCH, replace=False)
        xb, yb = X[idx], y[idx]

        # Forward: h = sum_p E[p, xb[:,p]]  (B, D)
        h = np.zeros((BATCH, D), np.float32)
        for p in range(d):
            h += E[p, xb[:, p]]

        logits = h @ W  # (B, C)
        grad_logits = softmax_grad(logits, yb)  # (B, C)

        # Backward
        grad_h = grad_logits @ W.T  # (B, D)
        grad_W = h.T @ grad_logits + L2 * W  # (D, C)

        grad_E = np.zeros_like(E)
        for p in range(d):
            # scatter-add grad_h to grad_E[p, xb[:,p]]
            np.add.at(grad_E[p], xb[:, p], grad_h)
        grad_E += L2 * E

        t += 1
        beta1, beta2 = 0.9, 0.999
        for param, grad, m, v in [(E, grad_E, mE, vE), (W, grad_W, mW, vW)]:
            m *= beta1; m += (1 - beta1) * grad
            v *= beta2; v += (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    magic = 0x4C523130  # "LR10"
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lr inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x4C523130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, sizeof(float) * D);
        for (int p = 0; p < d; p++) {
            const float *ep = E + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) h[j] += ep[j];
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int j = 0; j < D; j++) sum += h[j] * W[(long)j * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```