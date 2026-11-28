```hypothesis
Family: label-emb (embedding MLP). Learns a 256×E embedding per column, concatenates (14×E), then one ReLU hidden layer of size H and a 2-class output. Embeddings capture similarity among categories and ordinal bins better than one-hot linear; concatenation allows column interactions. Tuned on last 20% of train (E∈{4,8,16}, H∈{16,32,64}, lr∈{1e-3,3e-3}, L2∈{1e-4,1e-5}). If dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear, the model fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 12345

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, E, H, lr, l2, batch_size, epochs, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n, d = X.shape
    C = 2
    embeddings = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, np.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    params = [embeddings, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]
            yb = y[idx]
            batch = len(yb)
            embs = np.empty((batch, d, E), dtype=np.float32)
            for j in range(d):
                embs[:, j, :] = embeddings[j, xb[:, j], :]
            concat = embs.reshape(batch, d * E)
            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(batch), yb] -= 1.0
            grad_logits /= batch
            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = concat.T @ grad_hidden_pre
            grad_b1 = grad_hidden_pre.sum(axis=0)
            grad_concat = grad_hidden_pre @ W1.T
            grad_embs = grad_concat.reshape(batch, d, E)
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                np.add.at(grad_embeddings[j], xb[:, j], grad_embs[:, j, :])
            grad_embeddings += l2 * embeddings
            grad_W1 += l2 * W1
            grad_W2 += l2 * W2
            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return embeddings, W1, b1, W2, b2

def evaluate(X, y, embeddings, W1, b1, W2, b2):
    n, d = X.shape
    embs = np.empty((n, d, embeddings.shape[2]), dtype=np.float32)
    for j in range(d):
        embs[:, j, :] = embeddings[j, X[:, j], :]
    concat = embs.reshape(n, d * embeddings.shape[2])
    hidden = np.maximum(concat @ W1 + b1, 0)
    logits = hidden @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    param_grid = [
        {"E": 4, "H": 16, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 4, "H": 32, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 8, "H": 16, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 8, "H": 32, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 8, "H": 64, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 16, "H": 32, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 16, "H": 64, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 8, "H": 32, "lr": 3e-3, "l2": 1e-4, "batch": 64, "epochs": 30},
        {"E": 8, "H": 32, "lr": 1e-3, "l2": 1e-5, "batch": 64, "epochs": 30},
    ]
    best_acc = -1.0
    best_params = None
    for i, hp in enumerate(param_grid):
        seed = SEED + i + 1
        embeddings, W1, b1, W2, b2 = train_model(X_train, y_train, hp["E"], hp["H"], hp["lr"], hp["l2"], hp["batch"], hp["epochs"], seed)
        acc = evaluate(X_val, y_val, embeddings, W1, b1, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_params = hp
    print(f"Best validation accuracy: {best_acc:.4f} with params: {best_params}", file=sys.stderr)
    final_seed = SEED + 1000
    embeddings, W1, b1, W2, b2 = train_model(X, y, best_params["E"], best_params["H"], best_params["lr"], best_params["l2"], best_params["batch"], best_params["epochs"], final_seed)
    E, H = best_params["E"], best_params["H"]
    magic = 0x454D424C
    out = struct.pack("<5I", magic, d, E, H, C)
    out += embeddings.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1];
    int E = hd[2];
    int H = hd[3];
    int C = hd[4];
    if (C != 2) { fprintf(stderr, "unexpected classes\n"); return 3; }
    long expected = 20 + (long)d * 256 * E * 4 + (long)d * E * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *embeddings = (float *)(hd + 5);
    float *W1 = embeddings + (long)d * 256 * E;
    float *b1 = W1 + (long)d * E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + ((long)j * 256 + xi[j]) * E;
            float *dest = concat + j * E;
            for (int e = 0; e < E; e++) dest[e] = emb[e];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    free(concat);
    free(hidden);
    free(logits);
    return 0;
}
```