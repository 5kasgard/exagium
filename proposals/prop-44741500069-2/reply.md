```hypothesis
Family: label-embed. Learns per-position byte embeddings (size E) followed by an optional ReLU hidden layer (size H) and a linear output. Embeddings capture character similarities and reduce dimensionality vs one-hot, improving generalization on small data. If it fails to beat label-linear, it shows one-hot linear is already optimal for this task.
```
```python
import json, math, struct, sys

import numpy as np

# Tuning grid (embedding dim E, hidden size H)
CONFIGS = [
    {"E": 8, "H": 0},
    {"E": 16, "H": 0},
    {"E": 32, "H": 0},
    {"E": 8, "H": 16},
    {"E": 16, "H": 16},
    {"E": 32, "H": 32},
    {"E": 16, "H": 32},
    {"E": 32, "H": 64},
]

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # max samples per epoch


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy loss w.r.t logits (already averaged over batch)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_model(X, y, d, C, E, H, rng, epochs, batch_size):
    n = X.shape[0]
    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    if H:
        W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, C))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward
            # Embedding lookup: (bs, d, E)
            emb_batch = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)  # (bs, d, E)
            flat = emb_batch.reshape(bs, d * E)

            if H:
                hidden = flat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2
            else:
                logits = flat @ W + b

            # Loss gradient
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            # Backward
            grads = []
            if H:
                # Output layer
                g_W2 = hidden_relu.T @ g + L2 * W2
                g_b2 = g.sum(axis=0)
                # Hidden layer
                g_hidden = g @ W2.T
                g_hidden[hidden <= 0] = 0  # ReLU grad
                g_W1 = flat.T @ g_hidden + L2 * W1
                g_b1 = g_hidden.sum(axis=0)
                # Embeddings
                g_flat = g_hidden @ W1.T  # (bs, d*E)
                g_emb_batch = g_flat.reshape(bs, d, E).transpose(1, 0, 2)  # (d, bs, E)
                g_emb = np.zeros_like(emb)
                np.add.at(g_emb, (np.arange(d)[:, None], xb.T), g_emb_batch)
                g_emb += L2 * emb
                grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            else:
                g_W = flat.T @ g + L2 * W
                g_b = g.sum(axis=0)
                g_flat = g @ W.T
                g_emb_batch = g_flat.reshape(bs, d, E).transpose(1, 0, 2)
                g_emb = np.zeros_like(emb)
                np.add.at(g_emb, (np.arange(d)[:, None], xb.T), g_emb_batch)
                g_emb += L2 * emb
                grads = [g_emb, g_W, g_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    return params


def evaluate(X, y, d, C, E, H, params):
    n = X.shape[0]
    emb = params[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        bs = len(yb)
        emb_batch = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)
        flat = emb_batch.reshape(bs, d * E)
        if H:
            W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
            hidden = np.maximum(flat @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            W, b = params[1], params[2]
            logits = flat @ W + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1
    best_cfg = None
    best_params = None

    for cfg in CONFIGS:
        E, H = cfg["E"], cfg["H"]
        rng = np.random.default_rng(SEED + E * 1000 + H)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_model(X_train, y_train, d, C, E, H, rng, epochs, BATCH)
        acc = evaluate(X_val, y_val, d, C, E, H, params)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params

    # Retrain on full data with best config
    E, H = best_cfg["E"], best_cfg["H"]
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, d, C, E, H, rng, epochs, BATCH)

    # Pack weights
    emb = final_params[0]
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        magic = 0x454D4244  # "EMBD"
        header = struct.pack("<5i", magic, d, E, H, C)
        body = emb.astype("<f4").tobytes()
        body += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        body += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        magic = 0x454D4244
        header = struct.pack("<5i", magic, d, E, 0, C)
        body = emb.astype("<f4").tobytes()
        body += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * E * 4;          // embeddings
    if (H) {
        expected += (long)d * E * H * 4;        // W1
        expected += (long)H * 4;                // b1
        expected += (long)H * C * 4;            // W2
        expected += (long)C * 4;                // b2
    } else {
        expected += (long)d * E * C * 4;        // W
        expected += (long)C * 4;                // b
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    if (!H) {
        // then W1 is actually W, b1 is b
        W2 = NULL; b2 = NULL;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenate
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        if (H) {
            // hidden = relu(flat @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = flat @ W + b
            for (int c = 0; c < C; c++) {
                float sum = b1[c]; // b1 is actually b
                for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * C + c];
                logits[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```