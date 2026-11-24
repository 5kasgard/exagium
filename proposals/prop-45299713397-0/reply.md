```hypothesis
Family: label-emb-mlp — a tiny MLP with learned embeddings for categorical features.
Why: Adult has 8 categorical columns (each ≤255 categories) and 6 ordinal columns. One-hot expands categoricals to ~2000 dimensions; label-linear's 4096 multiply-adds/item are mostly sparse. Replacing one-hot with 4-dim embeddings (8×256×4 = 8 KiB) and a 16-unit ReLU layer cuts compute to ~672 multiply-adds/item while adding non-linear interactions. This should raise accuracy above label-linear's 0.8524 at lower CPU cost.
Fail condition: dev accuracy ≤ 0.8524 or instructions/item ≥ 719.
```

```python
"""label-emb-mlp: embedding + one-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weight format (little-endian):
  uint32 magic=0x454D424D ('EMBM'), int32 d, E, H, C
  uint8  feature_types[d]  (0=ordinal, 1=categorical)
  per feature:
    if categorical: float32[256*E] embedding table (row-major)
    if ordinal:     float32[2] mean, std
  float32 W_hidden[input_dim*H], b_hidden[H]
  float32 W_out[H*C], b_out[C]
where input_dim = (#categorical)*E + #ordinal.
"""
import json, math, struct, sys

import numpy as np

E, H = 4, 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Determine feature types and compute ordinal stats
    cat_idx = []
    ord_idx = []
    ord_mean = np.zeros(d, dtype=np.float32)
    ord_std = np.ones(d, dtype=np.float32)
    for j, t in enumerate(types):
        if t == "categorical":
            cat_idx.append(j)
        else:
            ord_idx.append(j)
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std()
            ord_mean[j] = np.float32(mu)
            ord_std[j] = np.float32(sd if sd > 0 else 1.0)

    n_cat = len(cat_idx)
    n_ord = len(ord_idx)
    input_dim = n_cat * E + n_ord

    # Initialize embeddings for categorical features
    emb = {}
    for j in cat_idx:
        # Xavier uniform for embedding: sqrt(6/(256+E))
        scale = math.sqrt(6.0 / (256 + E))
        emb[j] = (rng.uniform(-scale, scale, (256, E))).astype(np.float32)

    # Initialize hidden layer
    scale_h = math.sqrt(6.0 / (input_dim + H))
    W_hidden = (rng.uniform(-scale_h, scale_h, (input_dim, H))).astype(np.float32)
    b_hidden = np.zeros(H, dtype=np.float32)

    # Initialize output layer
    scale_o = math.sqrt(6.0 / (H + C))
    W_out = (rng.uniform(-scale_o, scale_o, (H, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [W_hidden, b_hidden, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Adam hyperparameters
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward: build input matrix
            Z = np.zeros((bs, input_dim), dtype=np.float32)
            col = 0
            for j in cat_idx:
                Z[:, col:col+E] = emb[j][xb[:, j]]
                col += E
            for j in ord_idx:
                Z[:, col] = (xb[:, j].astype(np.float32) - ord_mean[j]) / ord_std[j]
                col += 1

            # Hidden layer
            hidden_pre = Z @ W_hidden + b_hidden
            hidden = np.maximum(hidden_pre, 0)

            # Output layer
            logits = hidden @ W_out + b_out

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Gradients
            grad_W_out = hidden.T @ probs + L2 * W_out
            grad_b_out = probs.sum(axis=0)

            grad_hidden = probs @ W_out.T
            grad_hidden[hidden_pre <= 0] = 0.0

            grad_W_hidden = Z.T @ grad_hidden + L2 * W_hidden
            grad_b_hidden = grad_hidden.sum(axis=0)

            # Embedding gradients
            grad_emb = {}
            col = 0
            for j in cat_idx:
                # grad_Z for this embedding block: bs x E
                g = grad_hidden[:, col:col+E]
                # Accumulate per category
                ge = np.zeros((256, E), dtype=np.float32)
                np.add.at(ge, xb[:, j], g)
                grad_emb[j] = ge
                col += E
            # Ordinal gradients (already in grad_W_hidden for ordinal columns)

            grads = [grad_W_hidden, grad_b_hidden, grad_W_out, grad_b_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # Update embeddings
            for j in cat_idx:
                m, v = emb_adam[j]
                ge = grad_emb[j]
                m *= beta1; m += (1 - beta1) * ge
                v *= beta2; v += (1 - beta2) * ge * ge
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                emb[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    type_bytes = bytes(1 if t == "categorical" else 0 for t in types)

    out = bytearray()
    out += header
    out += type_bytes

    # Per-feature data
    for j in range(d):
        if types[j] == "categorical":
            out += emb[j].astype("<f4").tobytes()
        else:
            out += struct.pack("<2f", ord_mean[j], ord_std[j])

    # Hidden and output layers
    out += W_hidden.astype("<f4").tobytes()
    out += b_hidden.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index.
 * Weight format matches train.py. */
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
    unsigned char *ptr = wb;

    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = ptr[0] | ptr[1] << 8 | ptr[2] << 16 | ptr[3] << 24;
    if (magic != 0x454D424D) { fprintf(stderr, "bad magic\n"); return 3; }
    ptr += 4;
    int32_t d = *(int32_t*)ptr; ptr += 4;
    int32_t E = *(int32_t*)ptr; ptr += 4;
    int32_t H = *(int32_t*)ptr; ptr += 4;
    int32_t C = *(int32_t*)ptr; ptr += 4;

    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    unsigned char *feature_types = ptr; ptr += d;

    // Count categorical and ordinal
    int n_cat = 0, n_ord = 0;
    for (int i = 0; i < d; i++) {
        if (feature_types[i]) n_cat++; else n_ord++;
    }
    int input_dim = n_cat * E + n_ord;

    // Pointers to per-feature data
    float **cat_emb = malloc(n_cat * sizeof(float*));
    float *ord_mean = malloc(n_ord * sizeof(float));
    float *ord_std = malloc(n_ord * sizeof(float));
    int cat_ptr = 0, ord_ptr = 0;
    for (int i = 0; i < d; i++) {
        if (feature_types[i]) {
            cat_emb[cat_ptr++] = (float*)ptr;
            ptr += 256 * E * sizeof(float);
        } else {
            ord_mean[ord_ptr] = *(float*)ptr; ptr += 4;
            ord_std[ord_ptr] = *(float*)ptr; ptr += 4;
            ord_ptr++;
        }
    }

    // Hidden layer
    float *W_hidden = (float*)ptr; ptr += input_dim * H * sizeof(float);
    float *b_hidden = (float*)ptr; ptr += H * sizeof(float);
    // Output layer
    float *W_out = (float*)ptr; ptr += H * C * sizeof(float);
    float *b_out = (float*)ptr; ptr += C * sizeof(float);

    // Validate file size
    if (ptr - wb != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *input_vec = malloc(input_dim * sizeof(float));
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector
        int col = 0;
        int cat_idx = 0, ord_idx = 0;
        for (int j = 0; j < d; j++) {
            if (feature_types[j]) {
                const float *emb = cat_emb[cat_idx++];
                const float *row = emb + xi[j] * E;
                for (int e = 0; e < E; e++) input_vec[col++] = row[e];
            } else {
                float val = (xi[j] - ord_mean[ord_idx]) / ord_std[ord_idx];
                input_vec[col++] = val;
                ord_idx++;
            }
        }

        // Hidden layer: ReLU(input_vec * W_hidden + b_hidden)
        for (int h = 0; h < H; h++) {
            float sum = b_hidden[h];
            for (int id = 0; id < input_dim; id++) sum += input_vec[id] * W_hidden[id * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W_out[h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```