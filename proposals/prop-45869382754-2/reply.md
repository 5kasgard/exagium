```hypothesis
Family: label-emb-mlp (embedding + one ReLU hidden layer). Each of the 16 context bytes is mapped to a learned 8‑dim vector, concatenated (128‑dim), passed through a 32‑unit ReLU layer, then to 65 logits. This captures character similarities and positional context more efficiently than one‑hot linear, using ~39k parameters vs ~266k. Expect dev accuracy >0.3375 (label‑linear) with lower inference cost. If dev accuracy ≤0.3375, the family fails.
```
```python
"""Embedding + one ReLU hidden layer for next‑character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424D, int32 d=16, E=8, H=32, C=65;
float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension per byte
H = 32         # hidden units
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    embeddings = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward: gather embeddings
            emb_concat = np.zeros((bs, d * E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = embeddings[j, xb[:, j], :]

            h_pre = emb_concat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Backward
            g = softmax_grad(logits, yb)                    # (bs, C)
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T                                   # (bs, H)
            gh[h_pre <= 0] = 0.0
            grad_W1 = emb_concat.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            g_emb_concat = gh @ W1.T                        # (bs, d*E)

            # Scatter embedding gradients
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                np.add.at(grad_embeddings[j], xb[:, j], g_emb_concat[:, j*E:(j+1)*E])

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424D  # "EMBM" little‑endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += embeddings.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Embedding + one ReLU hidden layer inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Hidden layer: start with bias */
        for (int u = 0; u < H; u++) h[u] = b1[u];

        /* Add embeddings */
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((long)j * E + ee) * H + u];
                h[u] += sum;
            }
        }

        /* ReLU */
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        /* Output layer */
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu != 0.0f) {
                const float *w = W2 + (long)u * C;
                for (int c = 0; c < C; c++) sc[c] += hu * w[c];
            }
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```