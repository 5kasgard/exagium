```hypothesis
Family: label-embed-mlp. Uses shared byte embeddings (size 32) plus positional embeddings, concatenated across 16 positions, fed to a one-hidden-layer ReLU MLP (hidden 64) and a linear output. This shares statistical strength across positions while modeling interactions, unlike label-linear's position-specific one-hot weights. Expect higher dev accuracy than label-linear (0.3375) with moderate CPU cost (around 20k Ir/item). If dev accuracy ≤ 0.34 or cost > 50k Ir/item, the approach fails.
```

```python
"""label-embed-mlp: shared byte embeddings + positional embeddings -> concat -> ReLU hidden -> linear.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Hyperparameters chosen on last 20% tuning split."""
import json, math, struct, sys
import numpy as np

# Fixed architecture
E = 32          # embedding size
H = 64          # hidden size
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 128
EPOCHS_FULL = 20
EPOCHS_TUNE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    # byte_emb: 256 x E, pos_emb: d x E
    byte_emb = (rng.normal(0, 1, (256, E)) * math.sqrt(2.0 / 256)).astype(np.float32)
    pos_emb  = (rng.normal(0, 1, (d, E)) * math.sqrt(2.0 / d)).astype(np.float32)
    # W1: (d*E) x H, b1: H
    W1 = (rng.normal(0, 1, (d * E, H)) * math.sqrt(2.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C, b2: C
    W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return byte_emb, pos_emb, W1, b1, W2, b2

def forward(X, byte_emb, pos_emb, W1, b1, W2, b2):
    """X: (N, d) uint8. Returns logits (N, C) and hidden activations for backward."""
    N, d = X.shape
    # concatenated embeddings: (N, d*E)
    emb = np.empty((N, d * E), np.float32)
    for j in range(d):
        emb[:, j*E:(j+1)*E] = byte_emb[X[:, j]] + pos_emb[j]
    # hidden layer
    h = emb @ W1 + b1
    h_relu = np.maximum(h, 0)
    # output
    logits = h_relu @ W2 + b2
    return logits, emb, h_relu

def backward(X, y, emb, h_relu, logits, byte_emb, pos_emb, W1, W2, L2):
    N, d = X.shape
    C = W2.shape[1]
    # softmax gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N  # (N, C)

    # grad W2, b2
    gW2 = h_relu.T @ probs + L2 * W2
    gb2 = probs.sum(axis=0)

    # grad hidden relu
    gh = probs @ W2.T
    gh[h_relu == 0] = 0

    # grad W1, b1
    gW1 = emb.T @ gh + L2 * W1
    gb1 = gh.sum(axis=0)

    # grad emb (N, d*E)
    g_emb = gh @ W1.T  # (N, d*E)

    # grad byte_emb and pos_emb
    g_byte_emb = np.zeros_like(byte_emb)
    g_pos_emb = np.zeros_like(pos_emb)
    for j in range(d):
        g_emb_j = g_emb[:, j*E:(j+1)*E]
        # byte_emb gradient: scatter add
        np.add.at(g_byte_emb, X[:, j], g_emb_j)
        # pos_emb gradient: sum over batch
        g_pos_emb[j] = g_emb_j.sum(axis=0)

    return g_byte_emb, g_pos_emb, gW1, gb1, gW2, gb2

def train_loop(X, y, byte_emb, pos_emb, W1, b1, W2, b2, epochs, batch, lr, l2, rng, verbose=False):
    N = X.shape[0]
    # Adam state
    params = [byte_emb, pos_emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for ep in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch):
            idx = order[s:s+batch]
            Xb, yb = X[idx], y[idx]
            logits, emb, h_relu = forward(Xb, byte_emb, pos_emb, W1, b1, W2, b2)
            grads = backward(Xb, yb, emb, h_relu, logits, byte_emb, pos_emb, W1, W2, l2)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose:
            # compute train accuracy
            logits_all, _, _ = forward(X, byte_emb, pos_emb, W1, b1, W2, b2)
            acc = (logits_all.argmax(axis=1) == y).mean()
            print(f"epoch {ep+1}: train acc {acc:.4f}", file=sys.stderr)
    return byte_emb, pos_emb, W1, b1, W2, b2

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    N = X.shape[0]

    # split: last 20% for tuning
    split = int(N * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Hyperparameter search on validation set (small grid)
    best_val_acc = -1
    best_params = None
    for E_try in [16, 32]:
        for H_try in [32, 64]:
            for LR_try in [0.01, 0.03]:
                be, pe, W1, b1, W2, b2 = init_params(d, E_try, H_try, C, np.random.default_rng(SEED))
                be, pe, W1, b1, W2, b2 = train_loop(X_train, y_train, be, pe, W1, b1, W2, b2,
                                                    EPOCHS_TUNE, BATCH, LR_try, L2, rng)
                logits_val, _, _ = forward(X_val, be, pe, W1, b1, W2, b2)
                val_acc = (logits_val.argmax(axis=1) == y_val).mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = (E_try, H_try, LR_try)
    E, H, LR = best_params
    print(f"Chosen: E={E}, H={H}, LR={LR}, val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    rng = np.random.default_rng(SEED)
    byte_emb, pos_emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    byte_emb, pos_emb, W1, b1, W2, b2 = train_loop(X, y, byte_emb, pos_emb, W1, b1, W2, b2,
                                                    EPOCHS_FULL, BATCH, LR, L2, rng, verbose=True)

    # Write weights.bin
    # Header: magic, d, E, H, C
    magic = 0x454D4245  # 'EMBE'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Data in order: byte_emb (256*E), pos_emb (d*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    out = bytearray()
    out += header
    out += byte_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D4245), d, E, H, C (int32 each),
 * then float32 arrays: byte_emb[256][E], pos_emb[d][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (d != 16 || C < 1 || C > 256 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Expected file size
    long expected_floats = 256L * E + d * E + d * E * H + H + H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *byte_emb = (const float *)(hd + 5);
    const float *pos_emb = byte_emb + 256 * E;
    const float *W1 = pos_emb + d * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding: emb[pos*E + e] = byte_emb[xi[pos]*E + e] + pos_emb[pos*E + e]
        for (int p = 0; p < d; p++) {
            const float *be = byte_emb + xi[p] * E;
            const float *pe = pos_emb + p * E;
            float *dst = emb + p * E;
            for (int e = 0; e < E; e++) dst[e] = be[e] + pe[e];
        }

        // Hidden layer: ReLU(emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int j = 0; j < d * E; j++) acc += emb[j] * W1[j * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[h * C + c];
            scores[c] = acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb); free(hidden); free(scores);
    if (x) free(x);
    return 0;
}
```