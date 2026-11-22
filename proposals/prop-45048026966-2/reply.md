```hypothesis
Family: label-emb (per-position byte embeddings summed, optional ReLU hidden layer).
Why: The 16-byte context likely has compositional structure (n-grams) that embeddings can capture better than independent one-hot weights. Factorizing the linear layer into embeddings + output reduces parameters and acts as a regularizer, potentially improving generalization over label-linear (0.3375).
Failure: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear's ~7k Ir/item, the family does not justify its complexity.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters for grid search (tuned on last 20% of train)
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CONFIGS = [(32, 0), (32, 64), (64, 0), (64, 64), (128, 0), (128, 64)]  # (E, H)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits (B, C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)

def forward(emb, W1, b1, W2, b2, X, H):
    """X: (B, d) uint8. Returns logits (B, C) and cached values for backward."""
    B, d = X.shape
    E = emb.shape[2]
    # Sum embeddings: (B, E)
    emb_sum = np.zeros((B, E), dtype=np.float32)
    for j in range(d):
        emb_sum += emb[j, X[:, j], :]
    if H:
        hidden = np.maximum(emb_sum @ W1 + b1, 0, dtype=np.float32)
        logits = hidden @ W2 + b2
        return logits, (emb_sum, hidden)
    else:
        logits = emb_sum @ W2 + b2
        return logits, (emb_sum,)

def backward(dlogits, cache, X, emb, W1, W2, H, L2):
    """Compute gradients for all parameters."""
    B, d = X.shape
    E = emb.shape[2]
    C = dlogits.shape[1]
    grads = {}
    if H:
        emb_sum, hidden = cache
        # W2, b2
        grads['W2'] = hidden.T @ dlogits + L2 * W2
        grads['b2'] = dlogits.sum(axis=0)
        # hidden
        dhidden = dlogits @ W2.T
        dhidden[hidden <= 0] = 0
        # W1, b1
        grads['W1'] = emb_sum.T @ dhidden + L2 * W1
        grads['b1'] = dhidden.sum(axis=0)
        # emb_sum
        demb_sum = dhidden @ W1.T
    else:
        emb_sum, = cache
        grads['W2'] = emb_sum.T @ dlogits + L2 * W2
        grads['b2'] = dlogits.sum(axis=0)
        demb_sum = dlogits @ W2.T
    # emb gradients: scatter demb_sum to positions
    grads['emb'] = np.zeros_like(emb, dtype=np.float32)
    for j in range(d):
        # For each position j, add demb_sum[i] to emb[j, X[i, j], :]
        np.add.at(grads['emb'], (j, X[:, j], slice(None)), demb_sum)
    grads['emb'] += L2 * emb
    return grads

def adam_update(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for key in params:
        m, v = state[key]
        g = grads[key]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

def train_eval(X_train, y_train, X_val, y_val, E, H, seed):
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    C = 65
    # Initialize parameters
    emb = rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    params = {'emb': emb}
    if H:
        W1 = rng.uniform(-math.sqrt(6.0 / E), math.sqrt(6.0 / E), size=(E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.uniform(-math.sqrt(6.0 / H), math.sqrt(6.0 / H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params.update({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})
    else:
        W2 = rng.uniform(-math.sqrt(6.0 / E), math.sqrt(6.0 / E), size=(E, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params.update({'W2': W2, 'b2': b2})
    # Adam state
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, cache = forward(params['emb'],
                                    params.get('W1'), params.get('b1'),
                                    params['W2'], params['b2'],
                                    Xb, H)
            dlogits = softmax_cross_entropy_grad(logits, yb)
            grads = backward(dlogits, cache, Xb,
                             params['emb'],
                             params.get('W1'), params['W2'], H, L2)
            t += 1
            adam_update(params, grads, state, t, LR)
    # Evaluate on validation set
    logits_val, _ = forward(params['emb'],
                            params.get('W1'), params.get('b1'),
                            params['W2'], params['b2'],
                            X_val, H)
    pred = logits_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params

def write_weights(params, d, E, H, C, path):
    # Magic: 0x454D4245 ('EMBE')
    magic = 0x454D4245
    out = struct.pack("<5i", magic, d, E, H, C)
    out += params['emb'].astype("<f4").tobytes()
    if H:
        out += params['W1'].astype("<f4").tobytes()
        out += params['b1'].astype("<f4").tobytes()
        out += params['W2'].astype("<f4").tobytes()
        out += params['b2'].astype("<f4").tobytes()
    else:
        out += params['W2'].astype("<f4").tobytes()
        out += params['b2'].astype("<f4").tobytes()
    open(path, "wb").write(out)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    # Grid search
    best_acc, best_params, best_E, best_H = -1, None, None, None
    for E, H in CONFIGS:
        acc, params = train_eval(X_train, y_train, X_val, y_val, E, H, SEED)
        if acc > best_acc:
            best_acc, best_params, best_E, best_H = acc, params, E, H
    # Retrain on all data with best config
    print(f"Best config: E={best_E}, H={best_H}, tuning acc={best_acc:.4f}", file=sys.stderr)
    _, final_params = train_eval(X, y, X[:1], y[:1], best_E, best_H, SEED)  # dummy val
    write_weights(final_params, d, best_E, best_H, C, sys.argv[4])

if __name__ == "__main__":
    main()
```
```c
/* label-emb: per-position byte embeddings summed, optional ReLU hidden layer.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], E = (int)hd[2], H = (int)hd[3], C = (int)hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    // Expected size
    long expected = 20 + 4L * d * 256 * E;
    if (H) {
        expected += 4L * (E * H + H + H * C + C);
    } else {
        expected += 4L * (E * C + C);
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + (H ? E * H : 0);
    const float *W2 = b1 + (H ? H : 0);
    const float *b2 = W2 + (H ? H * C : E * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

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
            // Hidden layer: ReLU(emb_sum @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer: hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // Direct: emb_sum @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W2[(long)e * C + c];
                logits[c] = sum;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```