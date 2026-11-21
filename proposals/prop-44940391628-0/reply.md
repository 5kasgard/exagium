```hypothesis
Family: label-emb (embedding MLP). Each of the 14 byte features gets a learned 4-d embedding (256×4 table). Concatenated embeddings (56-d) feed a 32-unit ReLU hidden layer, then 2-class output. Embeddings capture non-linear ordinal/categorical structure more compactly than one-hot, so accuracy should exceed label-linear (0.8524) while inference cost stays near label-mlp16 (~2k Ir/item). Failure: dev accuracy ≤ 0.8524 or Ir/item > 3000.
```
```python
"""label-emb: embedding MLP for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 4          # embedding dimension per feature
H = 32         # hidden layer size
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
    d = X.shape[1]
    C = card["classes"]
    D = d * E
    rng = np.random.default_rng(SEED)

    # embedding tables: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = emb + [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward: gather embeddings
            Z = np.empty((bs, D), np.float32)
            for j in range(d):
                Z[:, j*E:(j+1)*E] = emb[j][xb[:, j]]

            a = Z @ W1 + b1
            h = np.maximum(a, 0, dtype=np.float32)
            logits = h @ W2 + b2

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # backward
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(axis=0)
            dh = probs @ W2.T
            da = dh * (a > 0)
            dW1 = Z.T @ da + L2 * W1
            db1 = da.sum(axis=0)
            dZ = da @ W1.T

            # embedding gradients (scatter add)
            demb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dZ[:, j*E:(j+1)*E])
                demb[j] += L2 * emb[j]

            grads = demb + [dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, E, H, C, then embeddings, W1, b1, W2, b2
    magic = 0x454D4231  # 'EMB1' little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    body = bytearray()
    for e in emb:
        body += e.astype("<f4").tobytes()
    body += W1.astype("<f4").tobytes()
    body += b1.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes()
    body += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E != 4 || H != 32) { fprintf(stderr, "unexpected dims\n"); return 3; }
    int D = d * E;
    long expected = 20L + 4L * ((long)d * 256 * E + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb[d];
    for (int j = 0; j < d; j++) {
        emb[j] = p;
        p += 256 * E;
    }
    const float *W1 = p; p += D * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb[j] + (size_t)xi[j] * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // hidden layer: ReLU(concat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < D; v++) sum += concat[v] * W1[(size_t)v * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(size_t)u * C + c];
            output[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```