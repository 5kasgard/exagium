```hypothesis
Family: label-embmlp — per-position byte embeddings summed, then a two-layer MLP (ReLU hidden, linear output).
Why: The 16-byte context has strong positional interactions; one-hot linear models ignore these. Embeddings compress the 256²⁶ sparse space, and a hidden layer captures non-linear combinations. Beats label-linear (0.3375) if dev accuracy > 0.35; otherwise the extra parameters are not justified.
```

```python
"""label-embmlp: per-position byte embeddings + 2-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 64          # embedding size
H = 128         # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
MAGIC = 0x4C424D45  # "EMBL" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)

    # Parameters (float32)
    embed = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    embed_bias = np.zeros(E, np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, embed_bias, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    n = X.shape[0]
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            # Sum embeddings for each position
            # xb: (B, d) -> for each position j, gather embed[j, xb[:, j], :] -> (B, E)
            emb_sum = embed_bias.copy()
            for j in range(d):
                emb_sum += embed[j, xb[:, j]]  # (B, E)
            hidden_pre = emb_sum @ W1 + b1
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # Backward
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_hidden = g_logits @ W2.T
            g_hidden[hidden_pre <= 0] = 0.0  # ReLU grad
            g_W1 = emb_sum.T @ g_hidden + L2 * W1
            g_b1 = g_hidden.sum(axis=0)
            g_emb_sum = g_hidden @ W1.T  # (B, E)

            # Embedding gradients
            g_embed = np.zeros_like(embed)
            g_embed_bias = g_emb_sum.sum(axis=0)
            for j in range(d):
                # scatter-add g_emb_sum[:, :] into g_embed[j, xb[:, j], :]
                np.add.at(g_embed[j], xb[:, j], g_emb_sum)
            g_embed += L2 * embed

            grads = [g_embed, g_embed_bias, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Order: embed [d][256][E], embed_bias [E], W1 [E][H], b1 [H], W2 [H][C], b2 [C]
    total_floats = d * 256 * E + E + E * H + H + H * C + C
    expected_bytes = 16 + 4 * total_floats
    out = struct.pack("<5i", MAGIC, d, E, H, C)
    out += embed.astype("<f4").tobytes()
    out += embed_bias.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    assert len(out) == expected_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * E + E + (long)E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *embed = p; p += (long)d * 256 * E;
    const float *embed_bias = p; p += E;
    const float *W1 = p; p += (long)E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // emb_sum = embed_bias
        memcpy(emb_sum, embed_bias, E * sizeof(float));
        // add embeddings for each position
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) emb_sum[ee] += e[ee];
        }

        // hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int ee = 0; ee < E; ee++) sum += emb_sum[ee] * W1[(long)ee * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```