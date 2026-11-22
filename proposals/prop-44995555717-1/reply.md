```hypothesis
Family: label-mlp2 (two-hidden-layer MLP with 32 and 16 ReLU units).  
Why: The 8×8 optdigits pixels have local spatial structure that a single linear layer or one hidden layer may not fully capture; two hidden layers can learn hierarchical features (edges → strokes → digits) while staying small. The model has ~3.5k parameters, still tiny for 2867 training samples, and inference cost is only ~2× label-linear.  
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-linear, the extra depth does not justify itself for this task.
```

```python
"""Two-hidden-layer MLP (32, 16 ReLU) for fixed-size byte-label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x4D4C5032, d, h1, h2, C; float32 T[d][256][h1], b1[h1],
W2[h1][h2], b2[h2], W3[h2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 32, 16
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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Encode each byte: ordinal -> z-scored single dimension
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape  # k == d for ordinal-only

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            g3 = softmax_grad(logits, yy)
            # backward
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(0)
            g2 = (g3 @ W3.T) * (a2 > 0)
            grad_W2 = h1.T @ g2 + L2 * W2
            grad_b2 = g2.sum(0)
            g1 = (g2 @ W2.T) * (a1 > 0)
            grad_W1 = z.T @ g1 + L2 * W1
            grad_b1 = g1.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build first-layer lookup tables T[j][v][u] = enc_j[v] * W1[j, u]
    off, T_list = 0, []
    for E in enc:
        # E: (256, 1) for ordinal
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, H1)

    # Pack weights
    header = struct.pack("<5i", 0x4D4C5032, d, H1, H2, C)
    body = (T.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes() +
            W3.astype("<f4").tobytes() +
            b3.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer MLP (32,16 ReLU) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax). */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], h1 = hd[2], h2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || h1 < 1 || h2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected_floats = (long)d * 256 * h1 + h1 + (long)h1 * h2 + h2 + (long)h2 * C + C;
    if (nw != 20 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * h1;
    const float *W2 = b1 + h1;
    const float *b2 = W2 + (long)h1 * h2;
    const float *W3 = b2 + h2;
    const float *b3 = W3 + (long)h2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * h1);
    float *acc2 = malloc(sizeof(float) * h2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, sizeof(float) * h1);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * h1;
            for (int u = 0; u < h1; u++) acc1[u] += t[u];
        }
        for (int u = 0; u < h1; u++) if (acc1[u] < 0.0f) acc1[u] = 0.0f;

        memcpy(acc2, b2, sizeof(float) * h2);
        for (int v = 0; v < h2; v++) {
            float sum = 0.0f;
            for (int u = 0; u < h1; u++) sum += acc1[u] * W2[(long)u * h2 + v];
            acc2[v] = sum;
        }
        for (int v = 0; v < h2; v++) if (acc2[v] < 0.0f) acc2[v] = 0.0f;

        memcpy(scores, b3, sizeof(float) * C);
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int v = 0; v < h2; v++) sum += acc2[v] * W3[(long)v * C + c];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```