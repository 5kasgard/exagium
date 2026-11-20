```hypothesis
Family: label-emb (embedding-based MLP with fused first layer). Adult has 8 categorical features with moderate cardinality; embeddings capture category similarities better than one-hot, and fusing embedding×W1 into a per-feature lookup table T[d][256][H] makes inference extremely fast (d×H adds + ReLU + H×C). With E=8, H=32 it should exceed label-linear (0.8524) and label-mlp16 (0.8591) accuracy while staying well below label-gbdt's 7570 Ir/item. Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 2325.
```
```python
"""label-emb: embedding MLP for tabular label tasks with fused first-layer lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Exports: magic 0x454D4231, d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E, H = 8, 32
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
    rng = np.random.default_rng(SEED)

    # embeddings: [d][256][E]
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # W1: [d*E][H]
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H][C]
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bs = len(yy)

            # forward: embed -> concat -> W1+b1 -> ReLU -> W2+b2
            x_emb = emb[np.arange(d)[:, None], xi.T]          # (d, bs, E)
            x_flat = x_emb.transpose(1, 0, 2).reshape(bs, d * E)  # (bs, d*E)
            h_pre = x_flat @ W1 + b1                          # (bs, H)
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2                              # (bs, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)     # (bs, C)

            # backward
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T * (h_pre > 0)
            grad_W1 = x_flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            grad_x_flat = grad_h @ W1.T                       # (bs, d*E)
            grad_x_emb = grad_x_flat.reshape(bs, d, E).transpose(1, 0, 2)  # (d, bs, E)

            # accumulate embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xi[:, j], grad_x_emb[j])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t = 1  # we'll use a global step counter
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                # bias correction (approximate with running t)
                # For simplicity and determinism, use a global step
                # We'll track t outside
                pass

    # The above loop needs a global step counter for Adam bias correction.
    # Let's rewrite the training loop properly with a global step.

    # Re-initialize for clean training with correct Adam
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bs = len(yy)

            x_emb = emb[np.arange(d)[:, None], xi.T]
            x_flat = x_emb.transpose(1, 0, 2).reshape(bs, d * E)
            h_pre = x_flat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T * (h_pre > 0)
            grad_W1 = x_flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            grad_x_flat = grad_h @ W1.T
            grad_x_emb = grad_x_flat.reshape(bs, d, E).transpose(1, 0, 2)

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xi[:, j], grad_x_emb[j])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fuse embeddings and W1 into T: T[j][v] = emb[j][v] @ W1[j*E:(j+1)*E]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j * E:(j + 1) * E]

    # Export
    magic = 0x454D4231  # "EMB1"
    out = struct.pack("<4i", magic, d, E, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class index.
 * Weights format: magic 0x454D4231, int32 d,E,H,C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc  = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += tj[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            for (int c = 0; c < C; c++) sc[c] += ah * W2[(long)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```