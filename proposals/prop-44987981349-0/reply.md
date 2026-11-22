```hypothesis
Family: label-fm (Factorization Machine with ordinal-aware encoding).
Why: Adult data has strong pairwise interactions (e.g., education×occupation) that linear models miss. FM captures all pairwise interactions via low-rank latent vectors (k=8) while keeping inference fast: O(d·k) per item. Ordinal columns use z-scored continuous features (preserving order) instead of one-hot, reducing parameters and improving generalization.
Failure signal: If dev accuracy ≤ label-linear (0.8524) or instructions/item ≥ label-linear (719), the family does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters (tuned on last 20% of train.in)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_K = [4, 8, 16]  # latent dimensions to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fm(X, y, types, k, rng, n_epochs, batch_size, lr, l2):
    n, d = X.shape
    # Precompute ordinal z-score tables on this training set
    ord_tables = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_tables.append(((np.arange(256) - mu) / sd).astype(np.float32))
        else:
            ord_tables.append(None)

    # Initialize parameters
    # For each column: if ordinal -> (w_j, v_j[k]); else -> (w_table[256], v_table[256, k])
    w_ord = np.zeros(d, np.float32)
    v_ord = np.zeros((d, k), np.float32)
    w_cat = [np.zeros(256, np.float32) for _ in range(d)]
    v_cat = [np.zeros((256, k), np.float32) for _ in range(d)]
    for j, t in enumerate(types):
        if t == "ordinal":
            w_ord[j] = 0.0
            v_ord[j] = rng.normal(0, 0.01 / math.sqrt(k), k).astype(np.float32)
        else:
            # categorical: initialize only seen categories? We'll init all small.
            w_cat[j] = rng.normal(0, 0.01, 256).astype(np.float32)
            v_cat[j] = rng.normal(0, 0.01 / math.sqrt(k), (256, k)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    params = []
    if any(t == "ordinal" for t in types):
        params.append(w_ord)
        params.append(v_ord)
    for j, t in enumerate(types):
        if t != "ordinal":
            params.append(w_cat[j])
            params.append(v_cat[j])
    params.append(bias.reshape(1))
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t_step = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            linear_sum = np.full(B, bias, np.float32)
            latent_sum = np.zeros((B, k), np.float32)
            latent_sq_sum = np.zeros((B, k), np.float32)

            for j, t in enumerate(types):
                xj = Xb[:, j]
                if t == "ordinal":
                    z = ord_tables[j][xj]                 # (B,)
                    linear_sum += w_ord[j] * z
                    vj = v_ord[j]                         # (k,)
                    contrib = vj * z[:, None]             # (B, k)
                    latent_sum += contrib
                    latent_sq_sum += contrib * contrib
                else:
                    w_vals = w_cat[j][xj]                 # (B,)
                    v_vals = v_cat[j][xj]                 # (B, k)
                    linear_sum += w_vals
                    latent_sum += v_vals
                    latent_sq_sum += v_vals * v_vals

            interaction = 0.5 * np.sum(latent_sum * latent_sum - latent_sq_sum, axis=1)
            logit = linear_sum + interaction
            p = sigmoid(logit)
            delta = p - yb                                # (B,)

            # Gradients
            grad_bias = delta.sum()
            grad_w_ord = np.zeros_like(w_ord)
            grad_v_ord = np.zeros_like(v_ord)
            grad_w_cat = [np.zeros_like(w_cat[j]) for j in range(d)]
            grad_v_cat = [np.zeros_like(v_cat[j]) for j in range(d)]

            for j, t in enumerate(types):
                xj = Xb[:, j]
                if t == "ordinal":
                    z = ord_tables[j][xj]
                    grad_w_ord[j] = np.sum(delta * z) + l2 * w_ord[j]
                    # grad_v_ord[j] = sum_b delta_b * (latent_sum_b - v_ord[j] * z_b)
                    # = (delta[:, None] * latent_sum).sum(axis=0) - v_ord[j] * (delta * z).sum()
                    grad_v_ord[j] = (delta[:, None] * latent_sum).sum(axis=0) - v_ord[j] * np.sum(delta * z)
                    grad_v_ord[j] += l2 * v_ord[j]
                else:
                    # Use np.add.at for categorical gradients
                    # For each category c, grad_w_cat[j][c] = sum_{b: xj[b]==c} delta_b + l2 * w_cat[j][c]
                    # grad_v_cat[j][c] = sum_{b: xj[b]==c} delta_b * (latent_sum_b - v_cat[j][c]) + l2 * v_cat[j][c]
                    # Vectorized with bincount
                    for f in range(k):
                        np.add.at(grad_v_cat[j][:, f], xj, delta * (latent_sum[:, f] - v_cat[j][xj, f]))
                    np.add.at(grad_w_cat[j], xj, delta)
                    grad_w_cat[j] += l2 * w_cat[j]
                    grad_v_cat[j] += l2 * v_cat[j]

            # Adam update
            grads = []
            if any(t == "ordinal" for t in types):
                grads.append(grad_w_ord)
                grads.append(grad_v_ord)
            for j, t in enumerate(types):
                if t != "ordinal":
                    grads.append(grad_w_cat[j])
                    grads.append(grad_v_cat[j])
            grads.append(np.array([grad_bias], np.float32))

            t_step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t_step)
                v_hat = v / (1 - 0.999 ** t_step)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack parameters for return
    model = {
        'bias': bias,
        'w_ord': w_ord,
        'v_ord': v_ord,
        'w_cat': w_cat,
        'v_cat': v_cat,
        'ord_tables': ord_tables
    }
    return model

def evaluate(model, X, y, types):
    n, d = X.shape
    linear_sum = np.full(n, model['bias'], np.float32)
    latent_sum = np.zeros((n, model['v_ord'].shape[1]), np.float32)
    latent_sq_sum = np.zeros_like(latent_sum)
    for j, t in enumerate(types):
        xj = X[:, j]
        if t == "ordinal":
            z = model['ord_tables'][j][xj]
            linear_sum += model['w_ord'][j] * z
            vj = model['v_ord'][j]
            contrib = vj * z[:, None]
            latent_sum += contrib
            latent_sq_sum += contrib * contrib
        else:
            linear_sum += model['w_cat'][j][xj]
            v_vals = model['v_cat'][j][xj]
            latent_sum += v_vals
            latent_sq_sum += v_vals * v_vals
    interaction = 0.5 * np.sum(latent_sum * latent_sum - latent_sq_sum, axis=1)
    logit = linear_sum + interaction
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Try candidate k values
    best_k = CANDIDATE_K[0]
    best_acc = -1.0
    for k in CANDIDATE_K:
        rng = np.random.default_rng(SEED + k * 1000)
        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        model = train_fm(X_train, y_train, types, k, rng, n_epochs, BATCH, LR, L2)
        acc = evaluate(model, X_tune, y_tune, types)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    # Retrain on all data with best_k
    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    model = train_fm(X, y, types, best_k, rng, n_epochs, BATCH, LR, L2)

    # Write weights.bin
    # Header: magic, d, k, C, types (d bytes)
    magic = 0x464D4C31  # "FML1"
    k = best_k
    header = struct.pack("<4i", magic, d, k, C)
    type_bytes = bytes(0 if t == "ordinal" else 1 for t in types)

    # Column data
    col_data = bytearray()
    for j, t in enumerate(types):
        if t == "ordinal":
            col_data += model['w_ord'][j].astype('<f4').tobytes()
            col_data += model['v_ord'][j].astype('<f4').tobytes()
            col_data += model['ord_tables'][j].astype('<f4').tobytes()
        else:
            col_data += model['w_cat'][j].astype('<f4').tobytes()
            col_data += model['v_cat'][j].astype('<f4').tobytes()

    bias_bytes = model['bias'].astype('<f4').tobytes()

    out = header + type_bytes + col_data + bias_bytes
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t k = *(int32_t*)(wb + 8);
    int32_t C = *(int32_t*)(wb + 12);
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad header dims\n"); return 3; }
    if (nw < 16 + d) { fprintf(stderr, "truncated weights\n"); return 3; }
    unsigned char *types = wb + 16;
    unsigned char *ptr = types + d;

    // Allocate column pointers
    float **w_ord = calloc(d, sizeof(float*));
    float **v_ord = calloc(d, sizeof(float*));
    float **z_table = calloc(d, sizeof(float*));
    float **w_cat = calloc(d, sizeof(float*));
    float **v_cat = calloc(d, sizeof(float*));

    for (int j = 0; j < d; j++) {
        if (types[j] == 0) { // ordinal
            if (ptr + 4 + 4*k + 4*256 > wb + nw) { fprintf(stderr, "truncated ordinal col %d\n", j); return 3; }
            w_ord[j] = (float*)ptr; ptr += 4;
            v_ord[j] = (float*)ptr; ptr += 4*k;
            z_table[j] = (float*)ptr; ptr += 4*256;
        } else { // categorical
            if (ptr + 4*256 + 4*256*k > wb + nw) { fprintf(stderr, "truncated categorical col %d\n", j); return 3; }
            w_cat[j] = (float*)ptr; ptr += 4*256;
            v_cat[j] = (float*)ptr; ptr += 4*256*k;
        }
    }
    if (ptr + 4 != wb + nw) { fprintf(stderr, "extra bytes after bias\n"); return 3; }
    float bias = *(float*)ptr;

    // Temporary buffers
    float *latent_sum = calloc(k, sizeof(float));
    float *latent_sq_sum = calloc(k, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float linear_sum = bias;
        memset(latent_sum, 0, k * sizeof(float));
        memset(latent_sq_sum, 0, k * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            if (types[j] == 0) { // ordinal
                float z = z_table[j][val];
                linear_sum += w_ord[j][0] * z;
                float *vj = v_ord[j];
                for (int f = 0; f < k; f++) {
                    float contrib = vj[f] * z;
                    latent_sum[f] += contrib;
                    latent_sq_sum[f] += contrib * contrib;
                }
            } else { // categorical
                linear_sum += w_cat[j][val];
                float *vrow = v_cat[j] + val * k;
                for (int f = 0; f < k; f++) {
                    float v = vrow[f];
                    latent_sum[f] += v;
                    latent_sq_sum[f] += v * v;
                }
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            interaction += latent_sum[f] * latent_sum[f] - latent_sq_sum[f];
        }
        interaction *= 0.5f;
        float logit = linear_sum + interaction;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```