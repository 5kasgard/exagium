```hypothesis
Family: label-embedmlp — shared byte embeddings + positional embeddings + one ReLU layer.
Why: The task predicts next character from 16 context bytes. label-linear uses 16×256×65 ≈ 266k separate weights, ignoring that the same byte value carries similar meaning across positions. This model learns a single 16-d embedding per byte value (shared across positions) plus a 16-d positional embedding, concatenates the 16 summed embeddings (256-d), then passes through a 32-unit ReLU layer to 65 classes. Parameters drop to ~15k, acting as strong regularization while capturing non-linear interactions. Precomputing the first layer into a (16,256,32) lookup table keeps inference fast.
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 15000.
```
```python
"""label-embedmlp: shared byte embeddings + positional embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D4245, d, D, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

D = 16          # embedding dimension
H = 32          # hidden units
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

    # Embeddings: byte_emb[256, D], pos_emb[d, D]
    byte_emb = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    pos_emb  = (rng.uniform(-1, 1, (d, D)) * math.sqrt(6.0 / (d + D))).astype(np.float32)

    # First layer: (d*D) -> H
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Second layer: H -> C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [byte_emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward: build concatenated embeddings (bs, d*D)
            # For each position j: byte_emb[xb[:,j]] + pos_emb[j]
            cat = np.empty((bs, d * D), np.float32)
            for j in range(d):
                cat[:, j*D:(j+1)*D] = byte_emb[xb[:, j]] + pos_emb[j]

            # Hidden layer
            h_pre = cat @ W1 + b1
            h = np.maximum(h_pre, 0)

            # Output
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = (probs / bs).astype(np.float32)  # (bs, C)

            # Backprop
            grad_h = grad_out @ W2.T                    # (bs, H)
            grad_h_pre = grad_h * (h_pre > 0)           # ReLU grad

            grad_W2 = h.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_W1 = cat.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)

            # Embedding gradients
            grad_cat = grad_h_pre @ W1.T                # (bs, d*D)
            grad_byte_emb = np.zeros_like(byte_emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                slice_j = grad_cat[:, j*D:(j+1)*D]
                np.add.at(grad_byte_emb, xb[:, j], slice_j)
                grad_pos_emb[j] += slice_j.sum(axis=0)

            grads = [grad_byte_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute first-layer lookup table T[d][256][H]
    # T[j, v, h] = sum_k (byte_emb[v,k] + pos_emb[j,k]) * W1[j*D + k, h]
    T = np.empty((d, 256, H), np.float32)
    for j in range(d):
        # (256, D) @ (D, H) -> (256, H)  but W1 is (d*D, H), slice per position
        W1_j = W1[j*D:(j+1)*D]  # (D, H)
        # byte_emb @ W1_j -> (256, H)
        # pos_emb[j] @ W1_j -> (H,) broadcast
        T[j] = byte_emb @ W1_j + pos_emb[j] @ W1_j

    # Pack weights: magic, d, D, H, C; T[d][256][H], b1[H], W2[H][C], b2[C]
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<5i", magic, d, D, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax, first max wins).
 * Weights format: int32 magic=0x454D4245, d, D, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || D < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Accumulate hidden pre-activations: h = b1 + sum_j T[j][xi[j]]
        memcpy(h, b1, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) h[u] += tj[u];
        }

        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        // Output: s = b2 + h @ W2
        memcpy(s, b2, sizeof(float) * C);
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) s[c] += hu * w2u[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```