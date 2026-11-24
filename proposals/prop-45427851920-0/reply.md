```hypothesis
Family: label-fm (Factorization Machine with per-feature embeddings).
Why: Adult is a classic tabular dataset where pairwise feature interactions matter (e.g., education×occupation). A factorization machine captures all pairwise interactions with O(d·k) parameters and inference cost, where k=8 is the embedding dimension. This should beat label-linear (0.8524) by modeling interactions while staying far cheaper than label-gbdt (7570 Ir/item).
Failure signal: If dev accuracy ≤ 0.8524 or Ir/item > 2000, the model is not a good trade-off.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
K = 8                # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 100

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
    # Parameters per class
    b = np.zeros(C, dtype=np.float32)
    W = np.zeros((C, d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(C, d, 256, K)).astype(np.float32)

    # Adam state
    mb, mW, mV = np.zeros_like(b), np.zeros_like(W), np.zeros_like(V)
    vb, vW, vV = np.zeros_like(b), np.zeros_like(W), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward pass: compute logits for both classes
            logits = np.zeros((B, C), dtype=np.float32)
            S = np.zeros((B, C, K), dtype=np.float32)
            S2 = np.zeros((B, C, K), dtype=np.float32)

            for c in range(C):
                # Linear term
                lin = np.zeros(B, dtype=np.float32)
                for j in range(d):
                    lin += W[c, j, xb[:, j]]
                # Interaction term
                S_c = np.zeros((B, K), dtype=np.float32)
                S2_c = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    v = V[c, j, xb[:, j], :]   # (B, K)
                    S_c += v
                    S2_c += v * v
                inter = 0.5 * np.sum(S_c * S_c - S2_c, axis=1)
                logits[:, c] = b[c] + lin + inter
                S[:, c, :] = S_c
                S2[:, c, :] = S2_c

            # Softmax and loss gradient
            max_logit = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B  # dL/dlogits

            # Backward pass
            t += 1
            # Bias
            gb = probs.sum(axis=0)
            mb *= beta1; mb += (1 - beta1) * gb
            vb *= beta2; vb += (1 - beta2) * gb * gb
            mhat = mb / (1 - beta1 ** t)
            vhat = vb / (1 - beta2 ** t)
            b -= LR * mhat / (np.sqrt(vhat) + eps) + LR * L2 * b

            for c in range(C):
                dc = probs[:, c]  # (B,)
                # W gradients
                for j in range(d):
                    idx_j = xb[:, j]
                    gW = np.zeros(256, dtype=np.float32)
                    np.add.at(gW, idx_j, dc)
                    mW[c, j] *= beta1; mW[c, j] += (1 - beta1) * gW
                    vW[c, j] *= beta2; vW[c, j] += (1 - beta2) * gW * gW
                    mhat = mW[c, j] / (1 - beta1 ** t)
                    vhat = vW[c, j] / (1 - beta2 ** t)
                    W[c, j] -= LR * mhat / (np.sqrt(vhat) + eps) + LR * L2 * W[c, j]

                # V gradients
                for j in range(d):
                    idx_j = xb[:, j]
                    v_j = V[c, j, idx_j, :]  # (B, K)
                    S_c = S[:, c, :]         # (B, K)
                    # dL/dV = dc[:, None] * (S_c - v_j)
                    gV = dc[:, None] * (S_c - v_j)  # (B, K)
                    # Accumulate per embedding row
                    for f in range(K):
                        gV_f = gV[:, f]
                        mV_f = mV[c, j, :, f]
                        vV_f = vV[c, j, :, f]
                        np.add.at(mV_f, idx_j, (1 - beta1) * gV_f)
                        mV_f *= beta1
                        np.add.at(vV_f, idx_j, (1 - beta2) * gV_f * gV_f)
                        vV_f *= beta2
                        mhat_f = mV_f / (1 - beta1 ** t)
                        vhat_f = vV_f / (1 - beta2 ** t)
                        V[c, j, :, f] -= LR * mhat_f / (np.sqrt(vhat_f) + eps) + LR * L2 * V[c, j, :, f]

    # Write weights
    # Format: magic, d, K, C; then per class: b, W (d*256), V (d*256*K)
    magic = 0x464D3132  # 'FM12'
    out = struct.pack("<4i", magic, d, K, C)
    for c in range(C):
        out += struct.pack("<f", b[c])
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected model dims\n"); return 3; }

    // Expected weight size: 4 ints + C * (1 float + d*256 floats + d*256*K floats)
    long expected = 16 + (long)C * (4 + 4L * d * 256 + 4L * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f;
        unsigned char best_class = 0;

        for (int c = 0; c < C; c++) {
            float logit = *p++;  // bias
            // Linear term
            for (int j = 0; j < d; j++) {
                logit += p[xi[j]];
            }
            p += 256;  // skip W for this feature
            // Interaction term
            for (int j = 0; j < d; j++) p += 256 * K;  // V will be accessed later
        }
        // We need to re-read the weights for interaction; easier to keep pointers.
        // Instead, let's restructure: keep base pointers.
        p = (const float *)(hd + 4);
        for (int c = 0; c < C; c++) {
            const float *bc = p++;  // bias
            const float *Wc = p; p += d * 256;
            const float *Vc = p; p += d * 256 * K;

            float logit = *bc;
            // Linear
            for (int j = 0; j < d; j++) {
                logit += Wc[j * 256 + xi[j]];
            }
            // Interaction
            for (int f = 0; f < K; f++) {
                float sum = 0.0f, sum2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = Vc[(j * 256 + xi[j]) * K + f];
                    sum += v;
                    sum2 += v * v;
                }
                logit += 0.5f * (sum * sum - sum2);
            }
            if (c == 0 || logit > best_score) {
                best_score = logit;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```