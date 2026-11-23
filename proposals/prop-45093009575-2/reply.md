```hypothesis
The label-embmlp family uses position-specific character embeddings (65 chars × 16 positions × 32 dims) followed by a single ReLU layer (128 units) to capture nonlinear n-gram interactions. This reduces the 4096 one-hot dimensions of label-linear to 512 dense features, improving generalization on the character-level language modeling task. If dev accuracy does not exceed 0.3375, the embedding bottleneck or single hidden layer is insufficient for this context length.
```
```python
"""label-embmlp: position-specific embeddings + 1 ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via internal tuning split search)
D = 32          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_byte_map(class_chars):
    """Map each byte value 0..255 to class index 0..C-1 (unknown -> 0)."""
    byte_to_class = np.zeros(256, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        b = ch.encode('utf-8')
        if len(b) == 1:
            byte_to_class[b[0]] = idx
    return byte_to_class

def prepare_data(card, train_in, train_key):
    X_bytes = np.frombuffer(b"".join(records(train_in)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_bytes = np.array([r[0] for r in records(train_key)], dtype=np.uint8)
    byte_to_class = build_byte_map(card["class_chars"])
    X_idx = byte_to_class[X_bytes]
    y_idx = byte_to_class[y_bytes]
    return X_idx, y_idx, byte_to_class

def init_params(L, V, D, H, C, rng):
    # Embeddings: (L, V, D) -> flatten to (L*V, D)
    E = (rng.uniform(-1, 1, (L * V, D)) * math.sqrt(6.0 / V)).astype(np.float32)
    # Input to hidden: (L*D, H)
    W1 = (rng.uniform(-1, 1, (L * D, H)) * math.sqrt(6.0 / (L * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Hidden to output: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [E, W1, b1, W2, b2]

def forward(params, X_idx, L, D, H):
    """Return (logits, hidden, concat) for backprop."""
    E, W1, b1, W2, b2 = params
    B = X_idx.shape[0]
    # Gather embeddings: for each position j, indices = j*V + X_idx[:,j]
    V = E.shape[0] // L
    concat = np.empty((B, L * D), np.float32)
    for j in range(L):
        indices = j * V + X_idx[:, j]
        concat[:, j*D:(j+1)*D] = E[indices]
    hidden_pre = concat @ W1 + b1
    hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
    logits = hidden @ W2 + b2
    return logits, hidden, concat, hidden_pre

def loss_and_grad(params, X_idx, y_idx, L, D, H, C):
    E, W1, b1, W2, b2 = params
    B = X_idx.shape[0]
    logits, hidden, concat, hidden_pre = forward(params, X_idx, L, D, H)
    # Softmax cross-entropy gradient
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y_idx] -= 1.0
    probs /= B
    # Gradients
    dW2 = hidden.T @ probs + L2 * W2
    db2 = probs.sum(axis=0)
    dh = probs @ W2.T
    dh[hidden_pre <= 0] = 0.0  # ReLU grad
    dW1 = concat.T @ dh + L2 * W1
    db1 = dh.sum(axis=0)
    dconcat = dh @ W1.T
    # Embedding gradients
    V = E.shape[0] // L
    dE = np.zeros_like(E)
    for j in range(L):
        indices = j * V + X_idx[:, j]
        np.add.at(dE, indices, dconcat[:, j*D:(j+1)*D])
    dE += L2 * E
    return [dE, dW1, db1, dW2, db2], logits

def accuracy(logits, y_idx):
    pred = logits.argmax(axis=1)
    return (pred == y_idx).mean()

def train_loop(X, y, params, L, D, H, C, epochs, batch_size, rng):
    n = X.shape[0]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            xb, yb = X[idx], y[idx]
            grads, _ = loss_and_grad(params, xb, yb, L, D, H, C)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, params, L, D, H):
    _, _, _, logits = forward(params, X, L, D, H)
    return accuracy(logits, y)

def main():
    card = json.load(open(sys.argv[1]))
    L = card["input"]["bytes"]
    C = card["classes"]
    V = C  # vocabulary size = number of classes (65)
    X_idx, y_idx, byte_to_class = prepare_data(card, sys.argv[2], sys.argv[3])
    n = X_idx.shape[0]
    split = int(0.8 * n)
    X_tr, X_tune = X_idx[:split], X_idx[split:]
    y_tr, y_tune = y_idx[:split], y_idx[split:]

    rng = np.random.default_rng(SEED)
    # Simple grid search on tuning split
    best_acc, best_params, best_D, best_H = -1, None, D, H
    for d in [16, 32, 64]:
        for h in [64, 128, 256]:
            params = init_params(L, V, d, h, C, rng)
            params = train_loop(X_tr, y_tr, params, L, d, h, C, epochs=20, batch_size=BATCH, rng=rng)
            # evaluate on tuning split
            logits, _, _, _ = forward(params, X_tune, L, d, h)
            acc = accuracy(logits, y_tune)
            if acc > best_acc:
                best_acc, best_params, best_D, best_H = acc, params, d, h
    print(f"Best tuning: D={best_D}, H={best_H}, acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best hyperparams
    params = init_params(L, V, best_D, best_H, C, rng)
    params = train_loop(X_idx, y_idx, params, L, best_D, best_H, C, epochs=MAX_EPOCHS, batch_size=BATCH, rng=rng)

    # Pack weights: magic, L, V, D, H, C, byte_map[256], then float32 arrays
    E, W1, b1, W2, b2 = params
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<6i", magic, L, V, best_D, best_H, C)
    byte_map = byte_to_class.tobytes()
    # Flatten arrays in order: E (L*V*D), W1 (L*D*H), b1 (H), W2 (H*C), b2 (C)
    body = (E.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + byte_map + body)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (context). Output record: 1 byte (class index 0..64). */
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 24 + 256) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)w;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int L = hd[1], V = hd[2], D = hd[3], H = hd[4], C = hd[5];
    if (L != 16 || V != 65 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    unsigned char *byte_map = w + 24;
    float *ptr = (float *)(w + 24 + 256);
    float *E = ptr; ptr += (long)L * V * D;
    float *W1 = ptr; ptr += (long)L * D * H;
    float *b1 = ptr; ptr += H;
    float *W2 = ptr; ptr += (long)H * C;
    float *b2 = ptr; ptr += C;
    if ((unsigned char *)ptr != w + nw) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * L * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != L) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, L); return 3; }
        /* Build concatenated embedding */
        for (int j = 0; j < L; j++) {
            int cls = byte_map[xi[j]];
            const float *emb = E + ((long)j * V + cls) * D;
            float *dst = concat + j * D;
            memcpy(dst, emb, D * sizeof(float));
        }
        /* Hidden layer: hidden = ReLU(concat * W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < L * D; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        /* Output layer: logits = hidden * W2 + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```