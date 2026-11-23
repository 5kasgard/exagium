```hypothesis
We propose a "label-linear-bigram" family that extends the linear table model with hashed bigram features. Character prediction benefits strongly from local context (adjacent characters); adding bigram features captures pairwise dependencies that unigrams miss while keeping inference fast (just 15 extra table lookups per item). The model uses the same Adam optimizer and per-position lookup tables as label-linear, but adds a shared bigram embedding table indexed by a simple hash of adjacent bytes. Hyperparameters (bigram bucket count B and hidden size H) are chosen on the tuning split. If dev accuracy does not beat label-linear (0.3375), it indicates bigram features are either too sparse or not helpful for this dataset.
```
```python
"""label-linear-bigram: linear model with unigram and hashed bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports weights as: magic 0x4247524D, d, H, C, B; float32 T_uni[d][256][W], b1[W], T_bi[B][W]; if H>0: W2[H][C], b2[C] (W=H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
B_GRID = [256, 512, 1024]
H_GRID = [0, 16, 32]

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


def compute_bigram_hashes(X, B):
    """X: (N, d) uint8 -> (N, d-1) int32 hashes"""
    return ((X[:, :-1].astype(np.uint32) << 8) + X[:, 1:].astype(np.uint32)) % B


def init_params(d, B, W, H, C, rng):
    """Initialize all parameters as float32 arrays."""
    # E_uni: (d, 256, W)
    limit = math.sqrt(6.0 / (1 + W))
    E_uni = rng.uniform(-limit, limit, (d, 256, W)).astype(np.float32)
    # E_bi: (B, W)
    E_bi = rng.uniform(-limit, limit, (B, W)).astype(np.float32)
    # W1: (W, H) if H>0 else (W, C)
    fan_in, fan_out = W, (H if H > 0 else C)
    limit1 = math.sqrt(6.0 / (fan_in + fan_out))
    W1 = rng.uniform(-limit1, limit1, (W, fan_out)).astype(np.float32)
    b1 = np.zeros(fan_out, dtype=np.float32)
    params = [E_uni, E_bi, W1, b1]
    if H > 0:
        limit2 = math.sqrt(6.0 / (H + C))
        W2 = rng.uniform(-limit2, limit2, (H, C)).astype(np.float32)
        b2 = np.zeros(C, dtype=np.float32)
        params.extend([W2, b2])
    return params


def train_model(X, y, hashes, d, B, H, C, epochs, batch_size, lr, l2, seed):
    """Train on given data, return learned parameters."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    W = H if H > 0 else C
    params = init_params(d, B, W, H, C, rng)
    E_uni, E_bi, W1, b1 = params[0], params[1], params[2], params[3]
    if H > 0:
        W2, b2 = params[4], params[5]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_cross_entropy_grad(logits, y_idx):
        # logits: (batch, C), y_idx: (batch,)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs.astype(np.float32) / len(y_idx)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            hb = hashes[idx]     # (b, d-1)
            bsz = len(idx)

            # Forward
            emb = np.zeros((bsz, W), dtype=np.float32)
            for j in range(d):
                emb += E_uni[j, Xb[:, j], :]
            for k in range(d - 1):
                emb += E_bi[hb[:, k], :]

            if H > 0:
                hidden_pre = emb @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                logits = hidden @ W2 + b2
            else:
                logits = emb @ W1 + b1

            # Backward
            dlogits = softmax_cross_entropy_grad(logits, yb)

            grads = [np.zeros_like(p) for p in params]
            gE_uni, gE_bi, gW1, gb1 = grads[0], grads[1], grads[2], grads[3]
            if H > 0:
                gW2, gb2 = grads[4], grads[5]

                gW2 = hidden.T @ dlogits
                gb2 = dlogits.sum(axis=0)
                dhidden = dlogits @ W2.T
                dhidden_pre = dhidden * (hidden_pre > 0)
                gW1 = emb.T @ dhidden_pre
                gb1 = dhidden_pre.sum(axis=0)
                demb = dhidden_pre @ W1.T
            else:
                gW1 = emb.T @ dlogits
                gb1 = dlogits.sum(axis=0)
                demb = dlogits @ W1.T

            # Distribute demb to E_uni and E_bi
            for j in range(d):
                np.add.at(gE_uni[j], Xb[:, j], demb)
            for k in range(d - 1):
                np.add.at(gE_bi, hb[:, k], demb)

            # L2 regularization on weights (not biases)
            gW1 += l2 * W1
            if H > 0:
                gW2 += l2 * W2
            gE_uni += l2 * E_uni
            gE_bi += l2 * E_bi

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params


def evaluate(X, y, hashes, d, B, H, C, params):
    """Return accuracy on given data."""
    E_uni, E_bi, W1, b1 = params[0], params[1], params[2], params[3]
    if H > 0:
        W2, b2 = params[4], params[5]
    n = X.shape[0]
    W = H if H > 0 else C
    correct = 0
    for i in range(n):
        emb = np.zeros(W, dtype=np.float32)
        for j in range(d):
            emb += E_uni[j, X[i, j], :]
        for k in range(d - 1):
            emb += E_bi[hashes[i, k], :]
        if H > 0:
            hidden = np.maximum(emb @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            logits = emb @ W1 + b1
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / n


def export_weights(card, d, H, C, B, params, out_path):
    """Write weights.bin in the format expected by infer.c."""
    E_uni, E_bi, W1, b1 = params[0], params[1], params[2], params[3]
    if H > 0:
        W2, b2 = params[4], params[5]
    W = H if H > 0 else C

    # Precompute folded tables
    T_uni = np.zeros((d, 256, W), dtype=np.float32)
    for j in range(d):
        T_uni[j] = E_uni[j] @ W1
    T_bi = E_bi @ W1

    magic = 0x4247524D  # 'BGRM'
    header = struct.pack("<5i", magic, d, H, C, B)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(T_uni.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(T_bi.astype("<f4").tobytes())
        if H > 0:
            f.write(W2.astype("<f4").tobytes())
            f.write(b2.astype("<f4").tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    N = X.shape[0]

    # Tuning split: last 20% in file order
    tune_size = N // 5
    train_idx = np.arange(N - tune_size)
    tune_idx = np.arange(N - tune_size, N)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Grid search on tuning split
    best_acc = -1.0
    best_B, best_H = B_GRID[0], H_GRID[0]
    for B in B_GRID:
        hashes_train = compute_bigram_hashes(X_train, B)
        hashes_tune = compute_bigram_hashes(X_tune, B)
        for H in H_GRID:
            n_train = len(train_idx)
            epochs = min(20, math.ceil(200000 / n_train))
            params = train_model(X_train, y_train, hashes_train, d, B, H, C,
                                 epochs, BATCH, LR, L2, SEED)
            acc = evaluate(X_tune, y_tune, hashes_tune, d, B, H, C, params)
            if acc > best_acc:
                best_acc = acc
                best_B, best_H = B, H

    # Retrain on all data with best hyperparameters
    B, H = best_B, best_H
    hashes_all = compute_bigram_hashes(X, B)
    n_all = N
    epochs_final = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    params = train_model(X, y, hashes_all, d, B, H, C,
                         epochs_final, BATCH, LR, L2, SEED)

    export_weights(card, d, H, C, B, params, sys.argv[4])


if __name__ == "__main__":
    main()
```
```c
/* label-linear-bigram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4247524D, d, H, C, B; float32 T_uni[d][256][W], b1[W], T_bi[B][W]; if H>0: W2[H][C], b2[C] (W=H if H>0 else C). */
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
    if (nw < 20 || hd[0] != 0x4247524D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], B = hd[4];
    int W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * W + W + (long)B * W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T_uni = (const float *)(hd + 5);
    const float *b1 = T_uni + (long)d * 256 * W;
    const float *T_bi = b1 + W;
    const float *W2 = H ? T_bi + (long)B * W : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // acc = b1
        for (int w = 0; w < W; w++) acc[w] = b1[w];

        // unigrams
        for (int j = 0; j < d; j++) {
            const float *t = T_uni + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        // bigrams
        for (int j = 0; j < d - 1; j++) {
            uint32_t hash = (xi[j] * 256u + xi[j + 1]) % (uint32_t)B;
            const float *t = T_bi + (long)hash * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }

        const float *sc = acc;
        if (H) {
            // ReLU
            for (int h = 0; h < H; h++) hidden[h] = acc[h] > 0.0f ? acc[h] : 0.0f;
            // output layer
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int h = 0; h < H; h++) {
                float hv = hidden[h];
                const float *w2 = W2 + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += hv * w2[c];
            }
            sc = scores;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```