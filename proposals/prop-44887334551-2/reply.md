```hypothesis
Family: label-embed (embedding + 1-hidden-layer MLP). Each of the 16 context bytes gets a learned 8-dim embedding; concatenated (128-dim) feeds a ReLU layer of 64 units, then a 65-class linear output. This captures per-position semantics and low-order interactions with far fewer parameters (≈45k) than label-linear's 266k one-hot weights, so it should generalize better on character prediction. If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear, the idea fails.
```

```python
"""label-embed: embedding + one-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[(d*E)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then train on all)
E = 8          # embedding dimension per byte
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: for each position, 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)

    # Hidden layer: input size d*E, output H
    in_dim = d * E
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H -> C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yy = y[idx]                # (B,)

            # Embedding lookup: (B, d, E)
            emb_lookup = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, E)
            flat = emb_lookup.reshape(len(idx), -1)                  # (B, d*E)

            # Forward
            h = flat @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Loss gradient
            g = softmax_grad(logits, yy)                 # (B, C)

            # Backprop
            g_W2 = h_relu.T @ g + L2 * W2
            g_b2 = g.sum(axis=0)
            g_h_relu = g @ W2.T
            g_h = g_h_relu * (h > 0)
            g_W1 = flat.T @ g_h + L2 * W1
            g_b1 = g_h.sum(axis=0)
            # Gradient w.r.t. embeddings
            g_flat = g_h @ W1.T                          # (B, d*E)
            g_emb_lookup = g_flat.reshape(len(idx), d, E) # (B, d, E)
            g_emb = np.zeros_like(emb)
            # Accumulate gradients per position and byte value
            for j in range(d):
                # xb[:, j] are byte values (0..255)
                np.add.at(g_emb[j], xb[:, j], g_emb_lookup[:, j, :])
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, H, C
    out = struct.pack("<5I", 0x454D424C, d, E, H, C)
    # embeddings: d * 256 * E floats
    out += emb.astype("<f4").tobytes()
    # W1, b1
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    # W2, b2
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
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
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20 + (long)d * 256 * E * 4 + (long)d * E * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((long)j * E + ee) * H + h];
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += hv * W2[(long)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```