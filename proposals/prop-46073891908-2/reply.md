```hypothesis
Family: label-embmlp — per-position byte embeddings (dim 8) feeding a 64-unit ReLU hidden layer, trained with Adam and exported as a fused first-layer lookup table (same binary format as label-linear). The low-rank factorization (16×(256×8 + 8×64) params) regularizes the large 16×256×64 first layer, improving generalization on this small dataset while keeping inference identical (one table lookup per byte + small MLP). If dev accuracy ≤ 0.34 or inference cost exceeds label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
D = 8          # embedding dimension per position
H = 64         # hidden layer size
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
    C = card["classes"]
    d = X.shape[1]          # 16
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # For each position: observed values, embeddings (256,D), W1 (D,H), obs_mask
    embeddings = []
    W1_list = []
    obs_masks = []
    for j in range(d):
        vals = np.unique(X[:, j])
        obs_mask = np.zeros(256, dtype=bool)
        obs_mask[vals] = True
        obs_masks.append(obs_mask)
        # Embedding matrix: 256 x D, unobserved rows stay zero
        emb = rng.uniform(-1, 1, (256, D)).astype(np.float32) * math.sqrt(6.0 / 256)
        emb[~obs_mask] = 0.0
        embeddings.append(emb)
        # First-layer weights: D x H
        w1 = rng.uniform(-1, 1, (D, H)).astype(np.float32) * math.sqrt(6.0 / D)
        W1_list.append(w1)

    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    # Adam state for all parameters: embeddings (16), W1_list (16), b1, W2, b2
    params = embeddings + W1_list + [b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            hidden = np.tile(b1, (B, 1))                 # (B, H)
            for j in range(d):
                emb_j = embeddings[j][xb[:, j]]          # (B, D)
                hidden += emb_j @ W1_list[j]             # (B, H)
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2               # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                      # (B, C)

            # Backward pass
            grad_W2 = hidden_relu.T @ grad_logits        # (H, C)
            grad_b2 = grad_logits.sum(axis=0)            # (C,)
            grad_hidden_relu = grad_logits @ W2.T        # (B, H)

            grad_hidden = grad_hidden_relu * (hidden > 0)  # (B, H)
            grad_b1 = grad_hidden.sum(axis=0)             # (H,)

            grad_embeddings = [np.zeros_like(emb) for emb in embeddings]
            grad_W1_list = [np.zeros_like(w1) for w1 in W1_list]
            for j in range(d):
                emb_j = embeddings[j][xb[:, j]]          # (B, D)
                grad_W1_list[j] = emb_j.T @ grad_hidden  # (D, H)
                grad_emb_batch = grad_hidden @ W1_list[j].T  # (B, D)
                np.add.at(grad_embeddings[j], xb[:, j], grad_emb_batch)
                # Zero gradients for unobserved values
                grad_embeddings[j][~obs_masks[j]] = 0.0

            # L2 regularization
            for j in range(d):
                grad_W1_list[j] += L2 * W1_list[j]
                grad_embeddings[j] += L2 * embeddings[j]
            grad_W2 += L2 * W2

            # Adam update
            t += 1
            all_grads = grad_embeddings + grad_W1_list + [grad_b1, grad_W2, grad_b2]
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export fused first-layer tables T[j][v] = embedding[v] @ W1_j  (256, H)
    T = []
    for j in range(d):
        T_j = embeddings[j] @ W1_list[j]                 # (256, H)
        T.append(T_j.astype(np.float32))
    T = np.stack(T)                                      # (d, 256, H)

    # Write weights.bin (identical format to label-linear with H>0)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding-MLP inference (same binary format as label-linear with H>0).
 * ./infer weights.bin in.rec N  → writes N one-byte class records to stdout. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```