```hypothesis
Propose label-emb-mlp: a factorized embedding MLP. Each of the 16 context bytes gets a learned 8-dim embedding (position-specific), concatenated into a 128-dim vector, passed through a 32-unit ReLU hidden layer, then a 65-class linear output. This captures non-linear positional interactions with ~39k parameters (vs linear's 266k), should exceed label-linear's 0.3375 accuracy. Failure: dev accuracy ≤ 0.34.
```

```python
"""label-emb-mlp: factorized embedding MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position byte embeddings (E=8) -> concat (d*E=128) -> ReLU hidden (H=32) -> linear output (C=65).
Adam lr=1e-3, L2=1e-5, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x454D4245, int32 d,E,H,C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension per byte
H = 32         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4245  # 'EMBE' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # read data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # initialize parameters
    # embeddings: (d, 256, E)
    embeddings = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # W1: (d*E, H), b1: (H,)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2: (H, C), b2: (C,)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p, dtype=np.float32) for p in params]
    v = [np.zeros_like(p, dtype=np.float32) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # ---- forward ----
            # embedding lookup: for each position j, gather embeddings
            emb = np.empty((B, d, E), dtype=np.float32)
            for j in range(d):
                emb[:, j, :] = embeddings[j, Xb[:, j], :]
            h0 = emb.reshape(B, d * E)          # (B, d*E)

            a1 = h0 @ W1 + b1                   # (B, H)
            h1 = np.maximum(a1, 0)              # ReLU

            logits = h1 @ W2 + b2               # (B, C)

            # softmax cross-entropy gradient
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B                    # (B, C)

            # ---- backward ----
            grad_W2 = h1.T @ grad_logits        # (H, C)
            grad_b2 = np.sum(grad_logits, axis=0)  # (C,)

            grad_h1 = grad_logits @ W2.T        # (B, H)
            grad_a1 = grad_h1 * (a1 > 0)        # ReLU derivative

            grad_W1 = h0.T @ grad_a1            # (d*E, H)
            grad_b1 = np.sum(grad_a1, axis=0)   # (H,)

            grad_h0 = grad_a1 @ W1.T            # (B, d*E)
            grad_emb = grad_h0.reshape(B, d, E) # (B, d, E)

            # L2 regularization gradients (not on biases)
            grad_W2 += L2 * W2
            grad_W1 += L2 * W1
            # embeddings L2
            grad_emb_list = []
            for j in range(d):
                g = np.zeros((256, E), dtype=np.float32)
                np.add.at(g, Xb[:, j], grad_emb[:, j, :])
                g += L2 * embeddings[j]
                grad_emb_list.append(g)

            grads = [np.stack(grad_emb_list, axis=0), grad_W1, grad_b1, grad_W2, grad_b2]

            # ---- Adam update ----
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # ---- write weights ----
    # flatten embeddings to (d, 256, E) in order
    out = struct.pack("<5i", MAGIC, d, E, H, C)
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
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D4245, int32 d,E,H,C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    long expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected_bytes); return 3; }

    const float *ptr = (const float *)(hd + 5);
    const float *embeddings = ptr; ptr += (long)d * 256 * E;
    const float *W1 = ptr; ptr += (long)d * E * H;
    const float *b1 = ptr; ptr += H;
    const float *W2 = ptr; ptr += (long)H * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *concat = (float *)malloc(sizeof(float) * d * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, emb, E * sizeof(float));
        }

        // hidden layer: concat (d*E) -> H
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) {
                sum += concat[k] * W1[(long)k * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f; // ReLU
        }

        // output layer: H -> C
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```