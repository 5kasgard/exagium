```hypothesis
Family: label-emb (position-specific embeddings summed then linear). 
Why: The task predicts the next character from 16 context bytes. Label-linear uses a huge one-hot weight matrix (16*256*65 params) which overfits; label-emb compresses each position/byte into a 32-dim embedding, sums them, then projects to 65 classes (16*256*32 + 32*65 params). This shares statistical strength across classes and positions, improving generalization on the small Tinyshakespeare dataset. 
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%, the embedding bottleneck loses too much information.
```

```python
"""label-emb: position-specific byte embeddings summed then linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: int32 magic 0x454D4231, d, E, C; float32 Emb[d][256][E], W[E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 32
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

    # Xavier initialization for embeddings and output layer
    scale_emb = math.sqrt(6.0 / (256 + E))
    scale_out = math.sqrt(6.0 / (E + C))
    Emb = (rng.uniform(-1, 1, (d, 256, E)) * scale_emb).astype(np.float32)
    W = (rng.uniform(-1, 1, (E, C)) * scale_out).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mEmb, vEmb = np.zeros_like(Emb), np.zeros_like(Emb)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward: sum embeddings
            h = np.zeros((B, E), np.float32)
            for j in range(d):
                h += Emb[j, Xb[:, j]]

            logits = h @ W + b
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B  # dL/dlogits

            # Gradients
            dW = h.T @ probs + L2 * W
            db = probs.sum(axis=0)
            dh = probs @ W.T  # (B, E)

            dEmb = np.zeros_like(Emb)
            for j in range(d):
                np.add.at(dEmb[j], Xb[:, j], dh)

            # Adam updates
            t += 1
            # Emb
            mEmb = beta1 * mEmb + (1 - beta1) * dEmb
            vEmb = beta2 * vEmb + (1 - beta2) * dEmb * dEmb
            mEmb_hat = mEmb / (1 - beta1 ** t)
            vEmb_hat = vEmb / (1 - beta2 ** t)
            Emb -= LR * mEmb_hat / (np.sqrt(vEmb_hat) + eps)
            # W
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW * dW
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)
            # b
            mb = beta1 * mb + (1 - beta1) * db
            vb = beta2 * vb + (1 - beta2) * db * db
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)

    # Pack weights: magic, d, E, C; Emb[d][256][E]; W[E][C]; b[C]
    out = struct.pack("<4i", 0x454D4231, d, E, C)
    out += Emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 4);
    const float *W = Emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb = Emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += emb[e];
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int e = 0; e < E; e++) sum += h[e] * W[(long)e * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```