```hypothesis
Family: label-embed-mlp. Uses per-position token embeddings (256×E) plus positional embeddings (16×E), averages them, then a ReLU hidden layer (H) and output layer. Captures local n-gram patterns via embeddings and position awareness, which linear models miss. Should beat label-linear (0.3375) on accuracy with similar inference cost (E=32, H=64 → ~15k params). If it fails to exceed 0.34, the bottleneck is likely insufficient capacity or the averaging loses too much sequence info.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning on the last 20% of train.in
GRID = [
    {"E": 16, "H": 32, "LR": 1e-3},
    {"E": 16, "H": 64, "LR": 1e-3},
    {"E": 32, "H": 32, "LR": 1e-3},
    {"E": 32, "H": 64, "LR": 1e-3},
    {"E": 16, "H": 32, "LR": 3e-3},
    {"E": 16, "H": 64, "LR": 3e-3},
    {"E": 32, "H": 32, "LR": 3e-3},
    {"E": 32, "H": 64, "LR": 3e-3},
]

SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES_PER_TRIAL = 100000   # for tuning trials
SAMPLES_FULL = 600000        for final training
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(E, H, C, rng):
    # token_emb: (256, E), pos_emb: (16, E), W1: (E, H), b1: (H,), W2: (H, C), b2: (C,)
    scale = math.sqrt(6.0 / (256 + E))
    token_emb = (rng.uniform(-1, 1, (256, E)) * scale).astype(np.float32)
    scale = math.sqrt(6.0 / (16 + E))
    pos_emb = (rng.uniform(-1, 1, (16, E)) * scale).astype(np.float32)
    scale = math.sqrt(6.0 / (E + H))
    W1 = (rng.uniform(-1, 1, (E, H)) * scale).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    scale = math.sqrt(6.0 / (H + C))
    W2 = (rng.uniform(-1, 1, (H, C)) * scale).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [token_emb, pos_emb, W1, b1, W2, b2]


def forward(X, params):
    token_emb, pos_emb, W1, b1, W2, b2 = params
    # X: (n, 16) uint8
    n = X.shape[0]
    # Embedding lookup: (n, 16, E)
    emb = token_emb[X] + pos_emb[None, :, :]  # broadcasting
    x = emb.mean(axis=1)  # (n, E)
    h = np.maximum(x @ W1 + b1, 0)  # ReLU
    logits = h @ W2 + b2
    return logits, h, x


def backward(X, y, params, h, x, L2):
    token_emb, pos_emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # Softmax gradient
    logits = h @ W2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # (n, C)

    # Gradients for W2, b2
    gW2 = h.T @ probs + L2 * W2
    gb2 = probs.sum(axis=0)

    # Backprop to h
    gh = probs @ W2.T  # (n, H)
    gh *= (h > 0)      # ReLU derivative

    # Gradients for W1, b1
    gW1 = x.T @ gh + L2 * W1
    gb1 = gh.sum(axis=0)

    # Backprop to x
    gx = gh @ W1.T  # (n, E)

    # Gradients for embeddings (average over 16 positions)
    # gx is (n, E), each position contributes gx / 16
    g_token = np.zeros_like(token_emb)
    g_pos = np.zeros_like(pos_emb)
    # For each position p, add gx / 16 to the corresponding token embedding
    for p in range(16):
        tokens = X[:, p]  # (n,)
        np.add.at(g_token, tokens, gx / 16.0)
        g_pos[p] += gx.sum(axis=0) / 16.0

    return [g_token, g_pos, gW1, gb1, gW2, gb2]


def accuracy(X, y, params):
    logits, _, _ = forward(X, params)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_loop(X, y, params, rng, samples, lr, l2, batch):
    n = X.shape[0]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            logits, h, x = forward(xb, params)
            grads = backward(xb, yb, params, h, x, l2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    best_acc = -1
    best_params = None
    best_hp = None

    for hp in GRID:
        rng = np.random.default_rng(SEED)
        params = init_params(hp["E"], hp["H"], C, rng)
        train_loop(X_tr, y_tr, params, rng, SAMPLES_PER_TRIAL, hp["LR"], L2, BATCH)
        acc = accuracy(X_val, y_val, params)
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            best_hp = hp
        print(f"Trial E={hp['E']} H={hp['H']} LR={hp['LR']} -> val acc={acc:.4f}")

    print(f"Best: E={best_hp['E']} H={best_hp['H']} LR={best_hp['LR']} acc={best_acc:.4f}")

    # Retrain on all data
    rng = np.random.default_rng(SEED)
    params = init_params(best_hp["E"], best_hp["H"], C, rng)
    # Use the best_params as initialization? Better to retrain from scratch with same seed.
    # We'll just continue training from the best_params (already trained on 80%).
    # But to be safe, we reinitialize and train on full data.
    params = init_params(best_hp["E"], best_hp["H"], C, rng)
    train_loop(X_all, y_all, params, rng, SAMPLES_FULL, best_hp["LR"], L2, BATCH)

    # Pack weights
    token_emb, pos_emb, W1, b1, W2, b2 = params
    d = X_all.shape[1]
    E = best_hp["E"]
    H = best_hp["H"]
    magic = 0x454D424C  # "LBME"
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        token_emb.astype("<f4").tobytes() +
        pos_emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + 16L * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *token_emb = p; p += 256 * E;
    const float *pos_emb = p; p += 16 * E;
    const float *W1 = p; p += E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // emb_sum = 0
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;

        // Sum token + pos embeddings
        for (int j = 0; j < d; j++) {
            const float *t_emb = token_emb + (long)xi[j] * E;
            const float *p_emb = pos_emb + (long)j * E;
            for (int e = 0; e < E; e++) emb_sum[e] += t_emb[e] + p_emb[e];
        }
        // Average
        for (int e = 0; e < E; e++) emb_sum[e] /= d;

        // Hidden layer: ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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