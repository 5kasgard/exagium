```hypothesis
Family: label-embmlp (embedding + one hidden layer MLP). The 16-byte context is treated as 16 categorical features; we learn an embedding per position per byte value, sum them, then pass through a ReLU hidden layer and a linear output. This captures interactions between positions that a linear model misses. The tinyshakespeare task has strong local character dependencies; embeddings allow sharing statistical strength across positions. If dev accuracy does not exceed label-linear (0.3375) by at least 0.01, the family does not justify its extra cost.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed seed for reproducibility
SEED = 42

# Hyperparameter grid to search on tuning split (last 20% of train.in)
# Each config: (embed_dim, hidden_dim, lr, l2, batch_size, max_epochs)
CONFIGS = [
    (64, 128, 3e-3, 1e-4, 256, 50),
    (32, 64, 3e-3, 1e-4, 256, 50),
    (128, 256, 1e-3, 1e-4, 256, 50),
]

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, embed_dim, hidden_dim, lr, l2, batch_size, max_epochs, rng_seed):
    n = X.shape[0]
    rng = np.random.default_rng(rng_seed)
    # Flattened embedding table: [d * 256, embed_dim]
    emb = rng.normal(0, 1.0 / math.sqrt(embed_dim), size=(d * 256, embed_dim)).astype(np.float32)
    # Hidden layer: [embed_dim, hidden_dim]
    W1 = rng.normal(0, math.sqrt(2.0 / embed_dim), size=(embed_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    # Output layer: [hidden_dim, C]
    W2 = rng.normal(0, math.sqrt(2.0 / hidden_dim), size=(hidden_dim, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    def softmax_cross_entropy_grad(logits, labels):
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(labels)), labels] -= 1.0
        return probs / len(labels)
    
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            x_batch = X[idx]  # [batch, d]
            y_batch = y[idx]
            batch_size_actual = len(idx)
            
            # Forward
            # indices for embedding lookup: (batch, d)
            indices = (np.arange(d, dtype=np.int32) * 256)[None, :] + x_batch  # shape (batch, d)
            # Sum embeddings: (batch, embed_dim)
            emb_sum = emb[indices].sum(axis=1)
            
            hidden = emb_sum @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2
            
            # Loss gradient
            grad_logits = softmax_cross_entropy_grad(logits, y_batch)  # [batch, C]
            
            # Backward
            grad_W2 = hidden_relu.T @ grad_logits + l2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden_relu = grad_logits @ W2.T
            grad_hidden = grad_hidden_relu * (hidden > 0)
            grad_W1 = emb_sum.T @ grad_hidden + l2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            # Embedding gradients: scatter-add grad_hidden to emb at indices
            grad_emb = np.zeros_like(emb)
            # Repeat grad_hidden for each position
            grad_hidden_repeated = np.repeat(grad_hidden, d, axis=0)  # (batch*d, embed_dim)
            np.add.at(grad_emb, indices.ravel(), grad_hidden_repeated)
            grad_emb += l2 * emb
            
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    
    return params

def evaluate(params, X, y, d, C):
    emb, W1, b1, W2, b2 = params
    embed_dim = emb.shape[1]
    n = X.shape[0]
    indices = (np.arange(d, dtype=np.int32) * 256)[None, :] + X
    emb_sum = emb[indices].sum(axis=1)
    hidden = emb_sum @ W1 + b1
    hidden_relu = np.maximum(hidden, 0)
    logits = hidden_relu @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]
    
    X = np.frombuffer(b"".join(records(train_in)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(train_key)], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    
    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    
    best_acc = -1.0
    best_config = None
    
    for config in CONFIGS:
        embed_dim, hidden_dim, lr, l2, batch_size, max_epochs = config
        params = train_model(X_train, y_train, d, C, embed_dim, hidden_dim, lr, l2, batch_size, max_epochs, SEED)
        acc = evaluate(params, X_tune, y_tune, d, C)
        if acc > best_acc:
            best_acc = acc
            best_config = config
    
    # Retrain best config on full dataset
    embed_dim, hidden_dim, lr, l2, batch_size, max_epochs = best_config
    final_params = train_model(X, y, d, C, embed_dim, hidden_dim, lr, l2, batch_size, max_epochs, SEED)
    
    # Save weights
    emb, W1, b1, W2, b2 = final_params
    # Magic number: 0x4D4C5031 ('MLP1')
    header = struct.pack("<5i", 0x4D4C5031, d, embed_dim, hidden_dim, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(weights_out, "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding + one hidden layer MLP for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic(0x4D4C5031), d, E, H, C; then float32 arrays:
 *   emb[d*256*E], W1[E*H], b1[H], W2[H*C], b2[C] */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
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
        // Hidden layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f; // ReLU
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
    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```