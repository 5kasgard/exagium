```hypothesis
Family: label-embed (per-position byte embeddings + small ReLU MLP).
Why: The task predicts the next character from 16 bytes of context. A linear model on one-hot bytes (label-linear) ignores interactions between positions. Learned embeddings share statistical strength across positions and characters, and a hidden ReLU layer captures non-linear compositions. This should improve accuracy over label-linear (0.3375) while keeping inference fast (embedding lookup + two small dense layers). If dev accuracy does not exceed label-linear at comparable or lower CPU cost, the hypothesis fails.
```

```python
"""label-embed: per-position byte embeddings + optional ReLU MLP for label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single thread, stdlib + numpy only.
Tunes embedding size E and hidden size H on last 20% of train.in, then trains on all.
Exports: magic 0x454D4244, d, E, H, C; float32 embeddings[d][256][E], W1, b1, (W2, b2 if H>0)."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
E_GRID = [16, 32]
H_GRID = [0, 32, 64]  # 0 = linear on embeddings
SEED = 12345
LR, L2, BATCH, MAX_EPOCHS_TUNE, MAX_EPOCHS_FULL = 3e-3, 1e-4, 64, 50, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, epochs, rng, lr=LR, l2=L2, batch=BATCH):
    n = X.shape[0]
    # Initialize parameters
    emb = (rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch_size = X_batch.shape[0]
        # Sum embeddings across positions
        emb_sum = np.zeros((batch_size, E), np.float32)
        for j in range(d):
            emb_sum += emb[j, X_batch[:, j]]
        if H:
            hidden = np.maximum(emb_sum @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            logits = emb_sum @ W1 + b1
        return emb_sum, (hidden if H else None), logits

    def backward(emb_sum, hidden, logits, y_batch, batch_size):
        # Softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch_size), y_batch] -= 1.0
        grad_logits = (probs / batch_size).astype(np.float32)

        grads = [np.zeros_like(p) for p in params]
        if H:
            # grad W2, b2
            grads[3] = hidden.T @ grad_logits + l2 * W2
            grads[4] = grad_logits.sum(axis=0)
            # grad hidden
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden <= 0] = 0.0  # ReLU
            # grad W1, b1
            grads[1] = emb_sum.T @ grad_hidden + l2 * W1
            grads[2] = grad_hidden.sum(axis=0)
            # grad emb_sum
            grad_emb_sum = grad_hidden @ W1.T
        else:
            # linear
            grads[1] = emb_sum.T @ grad_logits + l2 * W1
            grads[2] = grad_logits.sum(axis=0)
            grad_emb_sum = grad_logits @ W1.T

        # grad embeddings
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            # scatter add grad_emb_sum to grad_emb[j, X_batch[:, j]]
            idx = X_batch[:, j]
            np.add.at(grad_emb[j], idx, grad_emb_sum)
        grads[0] = grad_emb + l2 * emb
        return grads

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            X_batch, y_batch = X[idx], y[idx]
            emb_sum, hidden, logits = forward(X_batch)
            grads = backward(emb_sum, hidden, logits, y_batch, len(idx))
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, d, C, E, H, params):
    emb, W1, b1 = params[0], params[1], params[2]
    if H:
        W2, b2 = params[3], params[4]
    n = X.shape[0]
    correct = 0
    for i in range(n):
        emb_sum = np.zeros(E, np.float32)
        for j in range(d):
            emb_sum += emb[j, X[i, j]]
        if H:
            hidden = np.maximum(emb_sum @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            logits = emb_sum @ W1 + b1
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E, best_H = None, None

    for E in E_GRID:
        for H in H_GRID:
            # Use a derived seed for each config to keep deterministic
            config_rng = np.random.default_rng(SEED + E * 1000 + H)
            params = train_model(X_train, y_train, d, C, E, H, MAX_EPOCHS_TUNE, config_rng)
            acc = evaluate(X_tune, y_tune, d, C, E, H, params)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_E, best_H = E, H
            print(f"Tune E={E} H={H} acc={acc:.4f}", file=sys.stderr)

    print(f"Best: E={best_E} H={best_H} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    final_rng = np.random.default_rng(SEED + 999999)
    final_params = train_model(X, y, d, C, best_E, best_H, MAX_EPOCHS_FULL, final_rng)

    # Pack weights
    emb, W1, b1 = final_params[0], final_params[1], final_params[2]
    if best_H:
        W2, b2 = final_params[3], final_params[4]
    else:
        W2, b2 = None, None

    # Magic, d, E, H, C
    header = struct.pack("<5i", 0x454D4244, d, best_E, best_H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D4244, int32 d, E, H, C; float32 embeddings[d][256][E], W1[E][H], b1[H], (W2[H][C], b2[C] if H>0). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size
    long expected = 20
        + 4L * d * 256 * E
        + 4L * E * (H ? H : C)
        + 4L * (H ? H : C)
        + (H ? 4L * H * C + 4L * C : 0);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * (H ? H : C);
    const float *W2 = H ? b1 + H : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Allocate buffers
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        if (H) {
            // Hidden layer: ReLU(emb_sum * W1 + b1)
            for (int h = 0; h < H; h++) {
                float acc = b1[h];
                for (int e = 0; e < E; e++) acc += emb_sum[e] * W1[(long)e * H + h];
                hidden[h] = acc > 0.0f ? acc : 0.0f;
            }
            // Output layer: hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float acc = b2[c];
                for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
                scores[c] = acc;
            }
        } else {
            // Linear: emb_sum * W1 + b1 (W1 is E x C)
            for (int c = 0; c < C; c++) {
                float acc = b1[c];
                for (int e = 0; e < E; e++) acc += emb_sum[e] * W1[(long)e * C + c];
                scores[c] = acc;
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); if (hidden) free(hidden); free(scores);
    if (x) free(x); free(hd);
    return 0;
}
```