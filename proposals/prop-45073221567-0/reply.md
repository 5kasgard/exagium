```hypothesis
Family: label-emb-mlp — a shallow MLP with learned embeddings for categorical features (8-d) and z-scored ordinal features, one hidden ReLU layer (32 units). 
Why: One-hot encoding in label-linear/label-mlp16 creates high-dimensional sparse inputs; dense embeddings reduce dimensionality (70 vs. potentially hundreds) and capture category similarities, improving generalization. The model stays tiny (~19k parameters) so inference cost should be comparable to label-mlp16.
Failure signal: Dev accuracy ≤ 0.855 (not beating label-linear by a meaningful margin).
```
```python
"""label-emb-mlp: embeddings for categorical + ordinal z-score -> concat -> ReLU hidden -> linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
E = 8          # embedding size per categorical feature
H = 32         # hidden layer size
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

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.ms = [np.zeros_like(p) for p in params]
        self.vs = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            m = self.ms[i]
            v = self.vs[i]
            m *= self.beta1; m += (1 - self.beta1) * g
            v *= self.beta2; v += (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Split features
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    D = n_ord + n_cat * E

    # Ordinal normalization stats
    X_ord = X[:, ord_idx].astype(np.float32)
    ord_mean = X_ord.mean(axis=0)
    ord_std = X_ord.std(axis=0)
    ord_std[ord_std == 0] = 1.0

    # Initialize parameters
    # Categorical embeddings: one (256, E) per categorical feature
    cat_emb = [rng.normal(0, 0.01, size=(256, E)).astype(np.float32) for _ in range(n_cat)]
    # Hidden layer
    W1 = rng.normal(0, math.sqrt(2.0 / D), size=(D, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = cat_emb + [W1, b1, W2, b2]
    opt = Adam(params, LR)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s+BATCH]
            xb = X[batch_idx]
            yb = y[batch_idx]
            B = len(yb)

            # Forward
            xb_ord = xb[:, ord_idx].astype(np.float32)
            xb_ord_norm = (xb_ord - ord_mean) / ord_std  # (B, n_ord)

            cat_emb_batch = np.zeros((B, n_cat, E), dtype=np.float32)
            for i, feat in enumerate(cat_idx):
                cat_emb_batch[:, i, :] = cat_emb[i][xb[:, feat]]
            cat_emb_flat = cat_emb_batch.reshape(B, -1)  # (B, n_cat*E)

            concat = np.concatenate([xb_ord_norm, cat_emb_flat], axis=1)  # (B, D)
            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            # Softmax cross-entropy
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradients
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            dW2 = hidden.T @ dlogits + 2 * L2 * W2
            db2 = dlogits.sum(axis=0)

            dhidden = dlogits @ W2.T
            dhidden[hidden_pre <= 0] = 0

            dW1 = concat.T @ dhidden + 2 * L2 * W1
            db1 = dhidden.sum(axis=0)

            dconcat = dhidden @ W1.T  # (B, D)
            dxb_ord_norm = dconcat[:, :n_ord]
            dcat_emb_flat = dconcat[:, n_ord:].reshape(B, n_cat, E)

            # Gradients for categorical embeddings
            dcat_emb = [np.zeros_like(cat_emb[i]) for i in range(n_cat)]
            for i, feat in enumerate(cat_idx):
                np.add.at(dcat_emb[i], xb[:, feat], dcat_emb_flat[:, i, :])
                dcat_emb[i] += 2 * L2 * cat_emb[i]

            # Gradients for ordinal stats are not needed (fixed)
            grads = dcat_emb + [dW1, db1, dW2, db2]
            opt.step(grads)

    # Write weights.bin
    # Header: magic, d, E, H, C, types (d bytes)
    magic = 0x454D4244  # "EMBD"
    type_bytes = bytearray(1 if t == "categorical" else 0 for t in types)
    header = struct.pack("<5I", magic, d, E, H, C) + type_bytes

    # Ordinal stats: mean then std for each ordinal feature (in feature order)
    ord_stats = []
    for j in range(d):
        if types[j] == "ordinal":
            ord_stats.append(ord_mean[ord_idx.index(j)])
            ord_stats.append(ord_std[ord_idx.index(j)])
    ord_stats = np.array(ord_stats, dtype=np.float32)

    # Categorical embeddings: concatenated in feature order (only for categorical features)
    cat_emb_all = np.concatenate([cat_emb[i].ravel() for i in range(n_cat)]).astype(np.float32)

    # W1, b1, W2, b2
    W1_flat = W1.astype(np.float32).ravel()
    b1_flat = b1.astype(np.float32).ravel()
    W2_flat = W2.astype(np.float32).ravel()
    b2_flat = b2.astype(np.float32).ravel()

    out = header + ord_stats.tobytes() + cat_emb_all.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (6 ordinal, 8 categorical). Output: 1 byte class (0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Parse header
    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 14 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    unsigned char *types = wb + 20; // 5*4 = 20 bytes

    // Count ordinal/categorical
    int n_ord = 0, n_cat = 0;
    int ord_idx[14], cat_idx[14];
    for (int j = 0; j < d; j++) {
        if (types[j] == 0) ord_idx[n_ord++] = j;
        else cat_idx[n_cat++] = j;
    }
    int D = n_ord + n_cat * E;

    // Expected weight size
    long expected = 20 + d + 2L * n_ord * 4 + (long)n_cat * 256 * E * 4 + (long)D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    // Pointers into weight buffer
    float *p = (float *)(wb + 20 + d);
    float *ord_mean = p; p += n_ord;
    float *ord_std  = p; p += n_ord;
    float *cat_emb = p; p += (long)n_cat * 256 * E;
    float *W1 = p; p += (long)D * H;
    float *b1 = p; p += H;
    float *W2 = p; p += (long)H * C;
    float *b2 = p;

    // Allocate temp buffers
    float *concat = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concat vector
        int cidx = 0;
        // Ordinal part
        for (int o = 0; o < n_ord; o++) {
            int feat = ord_idx[o];
            float val = (xi[feat] - ord_mean[o]) / ord_std[o];
            concat[cidx++] = val;
        }
        // Categorical embeddings
        for (int ct = 0; ct < n_cat; ct++) {
            int feat = cat_idx[ct];
            unsigned char v = xi[feat];
            const float *emb = cat_emb + ((long)ct * 256 + v) * E;
            for (int e = 0; e < E; e++) concat[cidx++] = emb[e];
        }

        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < D; j++) sum += concat[j] * W1[(long)j * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
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

    free(concat); free(hidden); free(logits);
    free(wb); free(x);
    return 0;
}
```