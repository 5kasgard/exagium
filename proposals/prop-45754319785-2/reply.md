```hypothesis
Family: label-emb-mlp (embedding + one ReLU hidden layer). Why: the linear model treats each position independently; embeddings let the model learn distributed representations of bytes and their interactions across positions, while a hidden layer captures non-linear combinations. Precomputing position-byte hidden contributions into a lookup table keeps inference fast (16*H adds + H*C muls). If dev accuracy ≤ 0.3375 (label-linear) the extra complexity is not justified.
```

```python
"""label-emb-mlp: embedding + one ReLU hidden layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position byte embeddings (E=32) -> concat (16*E) -> hidden ReLU (H=128) -> 65-class softmax.
Trained with Adam (lr=1e-3, L2=1e-5, batch=64, epochs=50). Exports fused position-byte->hidden table."""
import json, math, struct, sys

import numpy as np

E = 32          # embedding size per position
H = 128         # hidden units
C = 65          # classes (from card)
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
EPOCHS = 50
MAGIC = 0x454D424D  # 'MBME' little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, targets):
    # logits: (B, C), targets: (B,) int
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(targets)), targets] -= 1.0
    return probs.astype(np.float32) / len(targets)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    assert d == 16
    C = card["classes"]
    assert C == 65

    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden weights: [d*E, H]
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output weights: [H, C]
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    flat_shapes = [p.size for p in params]

    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute embedding indices for speed: for each position, we will gather rows
    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]  # (B, d)
            yb = y[idx]

            # Forward
            # Embedding lookup: for each position j, get emb[j][Xb[:,j]] -> (B, E)
            # Concatenate along last axis -> (B, d*E)
            emb_concat = np.concatenate([emb[j, Xb[:, j]] for j in range(d)], axis=1)  # (B, d*E)
            hidden_pre = emb_concat @ W1 + b1  # (B, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2  # (B, C)

            # Loss gradient
            grad_logits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # Backward
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T  # (B, H)
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = emb_concat.T @ grad_hidden_pre + L2 * W1
            grad_b1 = grad_hidden_pre.sum(axis=0)
            # Gradient w.r.t. embeddings
            grad_emb_concat = grad_hidden_pre @ W1.T  # (B, d*E)
            grad_emb = np.zeros_like(emb)
            offset = 0
            for j in range(d):
                gj = grad_emb_concat[:, offset:offset + E]
                # scatter-add: for each sample, add gj[i] to grad_emb[j, Xb[i,j]]
                np.add.at(grad_emb[j], Xb[:, j], gj)
                offset += E
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, fuse embeddings and W1 into table T[d][256][H]
    # T[j, v, h] = sum_e emb[j, v, e] * W1[j*E + e, h]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j * E:(j + 1) * E]  # (256, H)

    # Write weights.bin
    # Header: magic, d, E, H, C (5 int32)
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # T: d * 256 * H float32
    T_bytes = T.astype("<f4").tobytes()
    # b1: H float32
    b1_bytes = b1.astype("<f4").tobytes()
    # W2: H * C float32
    W2_bytes = W2.astype("<f4").tobytes()
    # b2: C float32
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + T_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D424D), d, E, H, C (int32 each),
 *   T[d][256][H] (float32), b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        memcpy(hidden, b1, H * sizeof(float));
        // add T[j][xi[j]] for each position j
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += tj[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // scores = b2
        memcpy(scores, b2, C * sizeof(float));
        // scores += hidden * W2
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2h[c];
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```