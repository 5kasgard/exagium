```hypothesis
Family: label-embed (embedding-based MLP for fixed-byte categorical inputs).
Why: The input consists of 16 bytes from a known alphabet of 65 characters. A linear model uses one-hot encoding (16*256 features) which is high-dimensional and sparse. Learned embeddings (size E) per position compress this to 16*E dense features, capturing similarities between characters. A single hidden ReLU layer (size H) adds nonlinear interactions. This reduces parameters (from ~266k to ~20k) and can generalize better with limited data (24k train). The model is still fast at inference: embedding lookup + two small matrix multiplies.
Failure mode: If the best validation accuracy on the tuning split does not exceed label-linear's 0.3375, or if inference cost (instructions/item) exceeds label-linear's 7359.
```

```python
"""label-embed: embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (E, H). H=0 means linear on embeddings.
GRID = [(4, 0), (8, 0), (16, 0), (8, 32), (8, 64)]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SEARCH_EPOCHS = 30
FINAL_EPOCHS = 100
SAMPLES_PER_EPOCH = 600000  # not used directly, we use epochs


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, V, E, H, C, rng):
    """Initialize all parameters. Returns dict."""
    # embeddings per position: (d, V, E)
    emb = rng.normal(0, 0.01, size=(d, V, E)).astype(np.float32)
    if H > 0:
        W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return {'emb': emb, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'H': H}
    else:
        W_out = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        return {'emb': emb, 'W_out': W_out, 'b_out': b_out, 'H': 0}


def forward(params, indices, V, E, H, C):
    """Compute logits for a batch of indices (batch, d). Returns logits, cache."""
    d = indices.shape[1]
    batch = indices.shape[0]
    # Embedding lookup
    x = np.zeros((batch, d * E), dtype=np.float32)
    for p in range(d):
        x[:, p * E:(p + 1) * E] = params['emb'][p, indices[:, p], :]
    cache = {'x': x, 'indices': indices}
    if H > 0:
        a = x @ params['W1'] + params['b1']
        h = np.maximum(a, 0)
        logits = h @ params['W2'] + params['b2']
        cache['a'] = a
        cache['h'] = h
    else:
        logits = x @ params['W_out'] + params['b_out']
    return logits, cache


def loss_and_grad(params, cache, logits, y, V, E, H, C):
    """Compute loss and gradients for all parameters. Returns loss, grads dict."""
    batch = logits.shape[0]
    # Softmax cross-entropy
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(batch), y] + 1e-12).mean()

    dlogits = probs
    dlogits[np.arange(batch), y] -= 1.0
    dlogits /= batch

    grads = {}
    x = cache['x']
    indices = cache['indices']
    d = indices.shape[1]

    if H > 0:
        h = cache['h']
        a = cache['a']
        # Output layer
        grads['W2'] = h.T @ dlogits
        grads['b2'] = np.sum(dlogits, axis=0)
        dh = dlogits @ params['W2'].T
        da = dh * (a > 0)
        # Hidden layer
        grads['W1'] = x.T @ da
        grads['b1'] = np.sum(da, axis=0)
        dx = da @ params['W1'].T
    else:
        grads['W_out'] = x.T @ dlogits
        grads['b_out'] = np.sum(dlogits, axis=0)
        dx = dlogits @ params['W_out'].T

    # Embedding gradients
    demb = np.zeros_like(params['emb'])
    for p in range(d):
        np.add.at(demb[p], indices[:, p], dx[:, p * E:(p + 1) * E])
    grads['emb'] = demb

    # Add L2 regularization (on weights, not biases)
    for key in grads:
        if key in ('emb', 'W1', 'W2', 'W_out'):
            grads[key] += L2 * params[key]
    return loss, grads


def adam_update(params, grads, state, t):
    """In-place Adam update. state is dict of (m, v) for each param key."""
    for key in params:
        if key == 'H':
            continue
        p = params[key]
        g = grads[key]
        m, v = state[key]
        m *= 0.9
        m += 0.1 * g
        v *= 0.999
        v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)


def evaluate(params, X, y, V, E, H, C):
    """Compute accuracy on full dataset."""
    n = X.shape[0]
    correct = 0
    for i in range(0, n, BATCH):
        idx = X[i:i + BATCH]
        logits, _ = forward(params, idx, V, E, H, C)
        pred = np.argmax(logits, axis=1)
        correct += np.sum(pred == y[i:i + BATCH])
    return correct / n


def train_model(X_train, y_train, X_val, y_val, V, E, H, C, epochs, rng):
    """Train on X_train, optionally evaluate on X_val. Returns params, val_acc."""
    d = X_train.shape[1]
    params = init_params(d, V, E, H, C, rng)
    # Adam state
    state = {}
    for key in params:
        if key != 'H':
            state[key] = (np.zeros_like(params[key]), np.zeros_like(params[key]))
    n = X_train.shape[0]
    t = 0
    best_val_acc = -1
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_train[idx]
            yb = y_train[idx]
            logits, cache = forward(params, xb, V, E, H, C)
            loss, grads = loss_and_grad(params, cache, logits, yb, V, E, H, C)
            t += 1
            adam_update(params, grads, state, t)
        if X_val is not None:
            val_acc = evaluate(params, X_val, y_val, V, E, H, C)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    return params, best_val_acc


def save_weights(path, params, d, V, E, H, C, byte_to_class):
    """Write weights.bin in custom format."""
    # Header: magic, d, E, H, C
    magic = 0x454D4244  # "DBME"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Mapping: 256 bytes
    mapping = byte_to_class.astype(np.uint8).tobytes()
    # Embeddings: d * V * E floats (little-endian float32)
    emb = params['emb'].astype('<f4').tobytes()
    out = header + mapping + emb
    if H > 0:
        out += params['W1'].astype('<f4').tobytes()
        out += params['b1'].astype('<f4').tobytes()
        out += params['W2'].astype('<f4').tobytes()
        out += params['b2'].astype('<f4').tobytes()
    else:
        out += params['W_out'].astype('<f4').tobytes()
        out += params['b_out'].astype('<f4').tobytes()
    open(path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    class_chars = card["class_chars"]
    C = card["classes"]
    V = len(class_chars)  # 65

    # Build byte->class mapping
    byte_to_class = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx

    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    X = byte_to_class[X_bytes]
    assert np.all(X != 255), "Unknown byte in training data"

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)

    # Hyperparameter search
    best_val_acc = -1
    best_params = None
    best_E, best_H = None, None
    for E, H in GRID:
        print(f"Trying E={E}, H={H}...", file=sys.stderr)
        params, val_acc = train_model(X_train, y_train, X_val, y_val, V, E, H, C, SEARCH_EPOCHS, rng)
        print(f"  val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_E, best_H = E, H
            best_params = params

    print(f"Best: E={best_E}, H={best_H}, val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    final_params, _ = train_model(X, y, None, None, V, best_E, best_H, C, FINAL_EPOCHS, rng)

    # Save
    save_weights(sys.argv[4], final_params, d, V, best_E, best_H, C, byte_to_class)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)w;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    unsigned char *map = w + 20;
    float *emb = (float *)(w + 20 + 256);
    float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL, *W_out = NULL, *b_out = NULL;
    size_t emb_size = (size_t)d * 65 * E;
    if (H > 0) {
        W1 = emb + emb_size;
        b1 = W1 + (size_t)d * E * H;
        W2 = b1 + H;
        b2 = W2 + (size_t)H * C;
        long expected = 20 + 256 + 4 * (emb_size + (size_t)d * E * H + H + (size_t)H * C + C);
        if (nw != expected) { fprintf(stderr, "bad weights size (H>0)\n"); return 3; }
    } else {
        W_out = emb + emb_size;
        b_out = W_out + (size_t)d * E * C;
        long expected = 20 + 256 + 4 * (emb_size + (size_t)d * E * C + C);
        if (nw != expected) { fprintf(stderr, "bad weights size (H=0)\n"); return 3; }
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup
        for (int p = 0; p < d; p++) {
            unsigned char idx = map[xi[p]];
            if (idx >= 65) idx = 0; // fallback, should not happen
            const float *src = emb + ((size_t)p * 65 + idx) * E;
            float *dst = x_emb + p * E;
            memcpy(dst, src, E * sizeof(float));
        }
        if (H > 0) {
            // Hidden layer: x_emb (d*E) @ W1 (d*E x H) + b1
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int j = 0; j < d * E; j++) sum += x_emb[j] * W1[(size_t)j * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // Output layer: hidden (H) @ W2 (H x C) + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            // Linear: x_emb (d*E) @ W_out (d*E x C) + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int j = 0; j < d * E; j++) sum += x_emb[j] * W_out[(size_t)j * C + c];
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