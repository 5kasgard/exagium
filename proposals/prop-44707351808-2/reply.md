```hypothesis
Family: label-embed. Uses positional embeddings (16 positions × 256 byte values × E dimensions) summed into a single vector, followed by an optional ReLU hidden layer (size H) and a linear output layer (65 classes). This factorizes the large one-hot weight matrix, capturing character similarities and positional context with fewer parameters. Hyperparameters (E, H) are chosen on the tuning split (last 20% of train.in). Expected to exceed label-linear's 0.3375 dev accuracy; if dev accuracy ≤ 0.3375 or inference cost (Ir/item) exceeds label-linear's 7360, the hypothesis fails.
```

```python
"""label-embed: positional embedding + optional hidden layer for next-byte prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 12345
# Hyperparameter grid searched on tuning split (last 20% of train.in)
EMBED_DIMS = [8, 16, 32]
HIDDEN_SIZES = [0, 16, 32]  # 0 = no hidden layer
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # max samples per epoch (like label-linear)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, rng, n_samples=SAMPLES):
    n = X.shape[0]
    # Parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W_out = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(n_samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            h = np.zeros((len(yb), E), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            if H:
                a = h @ W1 + b1
                h_relu = np.maximum(a, 0)
                logits = h_relu @ W2 + b2
            else:
                logits = h @ W_out + b_out
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backward
            if H:
                grad_W2 = h_relu.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_h_relu = grad_logits @ W2.T
                grad_a = grad_h_relu * (a > 0)
                grad_W1 = h.T @ grad_a + L2 * W1
                grad_b1 = grad_a.sum(0)
                grad_h = grad_a @ W1.T
            else:
                grad_W_out = h.T @ grad_logits + L2 * W_out
                grad_b_out = grad_logits.sum(0)
                grad_h = grad_logits @ W_out.T
            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, scatter-add grad_h to grad_emb[j, xb[:, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_h)
            grad_emb += L2 * emb
            # Assemble grads
            if H:
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grads = [grad_emb, grad_W_out, grad_b_out]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def eval_model(X, y, params, d, C, H):
    n = X.shape[0]
    emb = params[0]
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W_out, b_out = params[1], params[2]
    correct = 0
    for s in range(0, n, 1024):
        xb, yb = X[s:s+1024], y[s:s+1024]
        h = np.zeros((len(yb), emb.shape[2]), np.float32)
        for j in range(d):
            h += emb[j, xb[:, j]]
        if H:
            a = h @ W1 + b1
            h_relu = np.maximum(a, 0)
            logits = h_relu @ W2 + b2
        else:
            logits = h @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_E, best_H = -1, None, None, None

    for E in EMBED_DIMS:
        for H in HIDDEN_SIZES:
            # Use a deterministic seed per config to ensure reproducibility
            cfg_rng = np.random.default_rng(SEED + E * 100 + H)
            params = train_model(train_X, train_y, d, C, E, H, cfg_rng)
            acc = eval_model(tune_X, tune_y, params, d, C, H)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H
                print(f"Tuning: E={E}, H={H}, acc={acc:.4f}", file=sys.stderr)

    print(f"Best: E={best_E}, H={best_H}, tune_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best hyperparameters
    final_rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, d, C, best_E, best_H, final_rng)

    # Write weights.bin
    # Format: magic(0x454D4245), d, E, H, C (int32), then embeddings (d*256*E floats),
    # then if H>0: W1 (E*H), b1 (H), W2 (H*C), b2 (C)
    # else: W_out (E*C), b_out (C)
    emb = final_params[0]
    out = struct.pack("<5i", 0x454D4245, d, best_E, best_H, C)
    out += emb.astype("<f4").tobytes()
    if best_H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W_out, b_out = final_params[1], final_params[2]
        out += W_out.astype("<f4").tobytes()
        out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D4245, int32 d, E, H, C; float32 emb[d][256][E];
 * if H>0: float32 W1[E][H], b1[H], W2[H][C], b2[C];
 * else: float32 W_out[E][C], b_out[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20;
    expected += 4L * d * 256 * E;
    if (H) {
        if (H <= 0) { fprintf(stderr, "bad H\n"); return 3; }
        expected += 4L * (E * H + H + H * C + C);
    } else {
        expected += 4L * (E * C + C);
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL, *W_out = NULL, *b_out = NULL;
    if (H) {
        W1 = emb + (long)d * 256 * E;
        b1 = W1 + (long)E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W_out = emb + (long)d * 256 * E;
        b_out = W_out + (long)E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // h = sum_j emb[j, xi[j]]
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e];
        }
        if (H) {
            // hidden = ReLU(h @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int e = 0; e < E; e++) sum += h[e] * W1[(long)e * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = h @ W_out + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int e = 0; e < E; e++) sum += h[e] * W_out[(long)e * C + c];
                logits[c] = sum;
            }
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```