```hypothesis
Family: label-embedmlp. A shallow neural network with per-position byte embeddings (dim=8) followed by a single ReLU hidden layer (64 units). This captures interactions between positions better than linear while keeping inference fast (lookup + two matvecs). Expected to beat label-linear (0.3375) on dev; if it doesn't, the added capacity may overfit or the embedding dimension is too small.
```

```python
"""label-embedmlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D4244, d, emb_dim, H, C; then float32 arrays:
  emb[d][256][emb_dim], W1[d*emb_dim][H], b1[H], W2[H][C], b2[C]."""
import json, math, os, struct, sys

import numpy as np

# Hard-coded hyperparameters (chosen on tuning split conceptually)
EMB_DIM = 8
H = 64
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 256
EPOCHS = 30

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    emb = rng.normal(0, 0.1, size=(d, 256, EMB_DIM)).astype(np.float32)
    fan_in = d * EMB_DIM
    W1 = rng.uniform(-math.sqrt(6/fan_in), math.sqrt(6/fan_in), size=(fan_in, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-math.sqrt(6/(H+C)), math.sqrt(6/(H+C)), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Pre-allocate buffers for speed
    flat_emb = np.empty((BATCH, fan_in), np.float32)
    hidden = np.empty((BATCH, H), np.float32)
    logits = np.empty((BATCH, C), np.float32)
    probs = np.empty((BATCH, C), np.float32)
    dlogits = np.empty((BATCH, C), np.float32)
    dh = np.empty((BATCH, H), np.float32)
    dh_pre = np.empty((BATCH, H), np.float32)
    demb_flat = np.empty((BATCH, fan_in), np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            bs = len(idx)
            xb = X[idx]
            yb = y[idx]

            # Forward: embed
            for j in range(d):
                flat_emb[:bs, j*EMB_DIM:(j+1)*EMB_DIM] = emb[j, xb[:, j]]
            # Hidden layer
            np.dot(flat_emb[:bs], W1, out=hidden[:bs])
            hidden[:bs] += b1
            np.maximum(hidden[:bs], 0, out=hidden[:bs])  # ReLU
            # Output layer
            np.dot(hidden[:bs], W2, out=logits[:bs])
            logits[:bs] += b2
            # Softmax + cross-entropy gradient
            max_logits = logits[:bs].max(axis=1, keepdims=True)
            np.exp(logits[:bs] - max_logits, out=probs[:bs])
            probs[:bs] /= probs[:bs].sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dlogits[:bs] = probs[:bs] / bs

            # Backward: output layer
            np.dot(hidden[:bs].T, dlogits[:bs], out=W2_grad := np.empty_like(W2))
            b2_grad = dlogits[:bs].sum(axis=0)
            # Hidden layer
            np.dot(dlogits[:bs], W2.T, out=dh[:bs])
            dh_pre[:bs] = dh[:bs] * (hidden[:bs] > 0)
            np.dot(flat_emb[:bs].T, dh_pre[:bs], out=W1_grad := np.empty_like(W1))
            b1_grad = dh_pre[:bs].sum(axis=0)
            # Embeddings
            np.dot(dh_pre[:bs], W1.T, out=demb_flat[:bs])
            # Accumulate embedding gradients per position
            emb_grad = np.zeros_like(emb)
            for j in range(d):
                # demb_flat[:bs, j*EMB_DIM:(j+1)*EMB_DIM] has gradients for position j
                # scatter-add into emb_grad[j, xb[:, j], :]
                np.add.at(emb_grad[j], xb[:, j], demb_flat[:bs, j*EMB_DIM:(j+1)*EMB_DIM])

            # Adam update with L2
            grads = [emb_grad, W1_grad, b1_grad, W2_grad, b2_grad]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if i != 2 and i != 4:  # no L2 on biases
                    gr += L2 * p
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5I", magic, d, EMB_DIM, H, C)
    out = bytearray(header)
    for arr in params:
        out += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index. */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long fan_in = (long)d * emb_dim;
    long expected = 20 + 4L * ( (long)d * 256 * emb_dim + fan_in * H + H + (long)H * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * emb_dim;
    const float *b1 = W1 + fan_in * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * fan_in);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * emb_dim;
            memcpy(flat + j * emb_dim, e, sizeof(float) * emb_dim);
        }
        // Hidden layer: ReLU(flat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < fan_in; v++) sum += flat[v] * W1[(long)v * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```