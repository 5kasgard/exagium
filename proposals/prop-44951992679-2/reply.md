```hypothesis
Family: label-embed (positional embeddings + optional one-hidden-layer MLP).
Why: The 16-byte context benefits from dense character embeddings that capture semantic similarity, reducing parameters vs. one-hot linear model. A small ReLU layer adds non-linearity for context interactions. Expected to exceed label-linear's 33.75% dev accuracy with similar or lower CPU cost due to smaller dense matrix multiplies.
Failure mode: If dev accuracy ≤ 33.75% or CPU cost significantly higher, the task may need longer-range sequence modeling (e.g., attention) or the linear model's high capacity with regularization is already optimal.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning
E_OPTIONS = [4, 8]
H_OPTIONS = [0, 16, 32]  # 0 means no hidden layer
LR_OPTIONS = [1e-3, 3e-3]
L2_OPTIONS = [1e-4, 1e-5]

# Training constants
SEED = 1
SAMPLES = 600000
BATCH = 64
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    embeddings = []
    for _ in range(d):
        scale = math.sqrt(6.0 / (256 + E))
        emb = rng.uniform(-scale, scale, (256, E)).astype(np.float32)
        embeddings.append(emb)
    if H:
        scale = math.sqrt(6.0 / (d*E + H))
        W1 = rng.uniform(-scale, scale, (d*E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        scale = math.sqrt(6.0 / (H + C))
        W2 = rng.uniform(-scale, scale, (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = embeddings + [W1, b1, W2, b2]
    else:
        scale = math.sqrt(6.0 / (d*E + C))
        W2 = rng.uniform(-scale, scale, (d*E, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = embeddings + [W2, b2]
    return params

def unpack_params(params, d, E, H, C):
    embeddings = params[:d]
    if H:
        W1, b1, W2, b2 = params[d], params[d+1], params[d+2], params[d+3]
    else:
        W1, b1 = None, None
        W2, b2 = params[d], params[d+1]
    return embeddings, W1, b1, W2, b2

def forward(X, embeddings, W1, b1, W2, b2, d, E, H):
    n = X.shape[0]
    embedded = np.zeros((n, d*E), np.float32)
    for j in range(d):
        embedded[:, j*E:(j+1)*E] = embeddings[j][X[:, j]]
    if H:
        hidden = embedded @ W1 + b1
        np.maximum(hidden, 0, out=hidden)
        logits = hidden @ W2 + b2
    else:
        logits = embedded @ W2 + b2
    return logits, embedded

def softmax_cross_entropy_grad(logits, y):
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_model(X, y, d, E, H, C, lr, l2, rng, epochs):
    n = X.shape[0]
    params = init_params(d, E, H, C, rng)
    embeddings, W1, b1, W2, b2 = unpack_params(params, d, E, H, C)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            logits, embedded = forward(Xb, embeddings, W1, b1, W2, b2, d, E, H)
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            if H:
                hidden = np.maximum(embedded @ W1 + b1, 0)
                dW2 = hidden.T @ g
                db2 = g.sum(0)
                dhidden = g @ W2.T
                dhidden *= (hidden > 0)
                dW1 = embedded.T @ dhidden
                db1 = dhidden.sum(0)
                dembedded = dhidden @ W1.T
            else:
                dW2 = embedded.T @ g
                db2 = g.sum(0)
                dembedded = g @ W2.T
            grad_embeddings = []
            for j in range(d):
                grad_emb = np.zeros((256, E), np.float32)
                np.add.at(grad_emb, Xb[:, j], dembedded[:, j*E:(j+1)*E])
                grad_embeddings.append(grad_emb)
            if H:
                dW2 += l2 * W2
                dW1 += l2 * W1
            else:
                dW2 += l2 * W2
            for j in range(d):
                grad_embeddings[j] += l2 * embeddings[j]
            grads = grad_embeddings
            if H:
                grads += [dW1, db1, dW2, db2]
            else:
                grads += [dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def eval_accuracy(X, y, params, d, E, H, C):
    logits, _ = forward(X, *unpack_params(params, d, E, H, C), d, E, H)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def write_weights(params, d, E, H, C, path):
    embeddings, W1, b1, W2, b2 = unpack_params(params, d, E, H, C)
    MAGIC = 0x44424D45  # 'EMBD' little-endian
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    emb_bytes = b"".join(emb.astype("<f4").tobytes() for emb in embeddings)
    if H:
        body = emb_bytes + W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        body = emb_bytes + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(header + body)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_acc = -1
    best_config = None
    best_params = None
    for i, E in enumerate(E_OPTIONS):
        for H in H_OPTIONS:
            for lr in LR_OPTIONS:
                for l2 in L2_OPTIONS:
                    config = (E, H, lr, l2)
                    rng = np.random.default_rng(SEED + i*1000 + H*100 + int(lr*1e4) + int(l2*1e6))
                    n_train = X_train.shape[0]
                    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
                    params = train_model(X_train, y_train, d, E, H, C, lr, l2, rng, epochs)
                    acc = eval_accuracy(X_val, y_val, params, d, E, H, C)
                    if acc > best_acc:
                        best_acc = acc
                        best_config = config
                        best_params = params
    print(f"Best config: E={best_config[0]}, H={best_config[1]}, lr={best_config[2]}, l2={best_config[3]}, val_acc={best_acc:.4f}", file=sys.stderr)
    E, H, lr, l2 = best_config
    rng = np.random.default_rng(SEED + 12345)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / N))
    final_params = train_model(X, y, d, E, H, C, lr, l2, rng, epochs)
    write_weights(final_params, d, E, H, C, sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
/* Embedding + optional one-hidden-layer MLP for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16); output is one byte (class index). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20;
    long emb_size = (long)d * 256 * E;
    if (H) {
        expected += 4 * (emb_size + (long)d * E * H + H + (long)H * C + C);
    } else {
        expected += 4 * (emb_size + (long)d * E * C + C);
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL;
    if (H) {
        W1 = emb + emb_size;
        b1 = W1 + (long)d * E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W2 = emb + emb_size;
        b2 = W2 + (long)d * E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *embedded = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(embedded, 0, sizeof(float) * d * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int kk = 0; kk < E; kk++) {
                embedded[j * E + kk] = e[kk];
            }
        }
        if (H)