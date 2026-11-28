```hypothesis
Family: label-emb (embedding + one-hidden-layer MLP). Replaces one-hot with learned per-position embeddings (dim E=16) to capture character similarity, then a ReLU hidden layer (H=64) and linear output. Should beat label-linear (0.3375) by learning distributed representations while keeping inference fast (small dense matmuls). If dev accuracy ≤0.34 or Ir/item >10k, the approach fails.
```

```python
"""label-emb: per-position embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
CONFIGS = [
    {"E": 8, "H": 32},
    {"E": 16, "H": 64},
    {"E": 32, "H": 128},
]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424C  # "EMBL"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, rng):
    # embeddings: (d, 256, E)
    W_emb = rng.normal(0, 0.1, size=(d, 256, E)).astype(np.float32)
    # hidden layer: (d*E, H)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer: (H, C)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return W_emb, W1, b1, W2, b2


def forward(X, W_emb, W1, b1, W2, b2):
    n, d = X.shape
    E = W_emb.shape[2]
    # embeddings lookup: (n, d, E)
    emb = W_emb[np.arange(d), X]  # (n, d, E)
    emb_flat = emb.reshape(n, d * E)  # (n, d*E)
    # hidden layer
    h = emb_flat @ W1 + b1
    h = np.maximum(h, 0)
    # output
    logits = h @ W2 + b2
    return logits, h, emb_flat


def loss_and_grad(X, y, W_emb, W1, b1, W2, b2):
    n = X.shape[0]
    logits, h, emb_flat = forward(X, W_emb, W1, b1, W2, b2)
    # softmax + cross-entropy
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # gradients
    grad_W2 = h.T @ probs + L2 * W2
    grad_b2 = probs.sum(axis=0)
    grad_h = probs @ W2.T
    grad_h *= (h > 0)
    grad_W1 = emb_flat.T @ grad_h + L2 * W1
    grad_b1 = grad_h.sum(axis=0)
    grad_emb_flat = grad_h @ W1.T  # (n, d*E)
    grad_emb = grad_emb_flat.reshape(n, X.shape[1], W_emb.shape[2])  # (n, d, E)
    # accumulate embedding gradients
    grad_W_emb = np.zeros_like(W_emb)
    d = X.shape[1]
    for j in range(d):
        np.add.at(grad_W_emb[j], X[:, j], grad_emb[:, j])
    grad_W_emb += L2 * W_emb
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    return loss, (grad_W_emb, grad_W1, grad_b1, grad_W2, grad_b2)


def train_model(X, y, config, seed, epochs):
    d = X.shape[1]
    C = 65
    E, H = config["E"], config["H"]
    rng = np.random.default_rng(seed)
    W_emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [W_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            _, grads = loss_and_grad(Xb, yb, W_emb, W1, b1, W2, b2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def accuracy(X, y, params):
    W_emb, W1, b1, W2, b2 = params
    logits, _, _ = forward(X, W_emb, W1, b1, W2, b2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(d, E, H, C, params):
    W_emb, W1, b1, W2, b2 = params
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    body = (
        W_emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    return header + body


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    best_acc, best_config, best_params = -1, None, None
    for cfg in CONFIGS:
        params = train_model(X_train, y_train, cfg, SEED, epochs)
        acc = accuracy(X_tune, y_tune, params)
        if acc > best_acc:
            best_acc, best_config, best_params = acc, cfg, params

    # Retrain on full data with best config
    final_params = train_model(X, y, best_config, SEED, epochs)
    out = pack_weights(d, best_config["E"], best_config["H"], C, final_params)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: per-position embeddings + one ReLU layer.
 * ./infer weights.bin in.rec N
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_emb = (const float *)(hd + 5);
    const float *W1 = W_emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *src = W_emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + (long)j * E, src, E * sizeof(float));
        }

        // hidden layer: ReLU(W1^T * emb_concat + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int k = 0; k < d * E; k++) sum += emb_concat[k] * W1[(long)k * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: W2^T * hidden + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            output[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```