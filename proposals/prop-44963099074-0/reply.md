```hypothesis
Family: label-fm (Factorization Machine with per-feature embeddings).
Why: Adult has 14 mixed ordinal/categorical features; FM captures pairwise interactions via low-rank embeddings, improving over linear models without the cost of deep nets. The 256-bin encoding per feature fits naturally into per-feature embedding tables.
Failure: If dev accuracy ≤ label-linear + 0.01 (≤0.8624) or inference cost > 2× label-linear, the family is not justified.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = {
    'k': [4, 8, 16],
    'lr': [0.01, 0.03, 0.1],
    'l2': [1e-4, 1e-3],
    'epochs': [20, 40, 60],
}
BATCH = 256

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X, bias, linear, emb, k):
    # X: (n, d) uint8
    n, d = X.shape
    # Linear part
    lin = bias + np.sum(linear[np.arange(d), X.T], axis=0)  # (n,)
    # Interaction part via (sum^2 - sum_sq)/2
    # emb shape: (d, 256, k)
    # Get embeddings for all features: (n, d, k)
    E = emb[np.arange(d), X.T].transpose(1, 0, 2)  # (n, d, k)
    sum_emb = np.sum(E, axis=1)  # (n, k)
    sum_sq_emb = np.sum(E * E, axis=1)  # (n, k)
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)  # (n,)
    return lin + inter

def train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, rng):
    n, d = X_tr.shape
    # Initialize parameters
    bias = 0.0
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, k)) * np.sqrt(2.0 / k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val = -1
    best_params = None
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            logits = fm_forward(xb, bias, linear, emb, k)
            p = sigmoid(logits)
            # Gradient
            grad = (p - yb).astype(np.float32)  # (batch,)
            # Bias
            g_bias = grad.mean()
            # Linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad)
            g_lin /= len(yb)
            # Embedding
            g_emb = np.zeros_like(emb)
            # Efficient gradient for FM interaction part
            # For each sample, we need sum_emb and per-feature emb
            E = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (batch, d, k)
            sum_emb = np.sum(E, axis=1)  # (batch, k)
            for j in range(d):
                v_j = E[:, j, :]  # (batch, k)
                # d(inter)/d v_j = sum_emb - v_j
                g_vj = (sum_emb - v_j) * grad[:, None]  # (batch, k)
                # Accumulate per feature value
                vals = xb[:, j]
                for b in range(len(yb)):
                    g_emb[j, vals[b]] += g_vj[b]
            g_emb /= len(yb)
            # Add L2 regularization gradients
            g_lin += l2 * linear
            g_emb += l2 * emb
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # Validation
        val_logits = fm_forward(X_val, bias, linear, emb, k)
        val_pred = (sigmoid(val_logits) > 0.5).astype(np.int64)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_params = (bias.copy(), linear.copy(), emb.copy())
    return best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_val = -1
    best_cfg = None
    best_params = None
    # Grid search
    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for epochs in GRID['epochs']:
                    params = train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, rng)
                    bias, linear, emb = params
                    val_logits = fm_forward(X_val, bias, linear, emb, k)
                    val_pred = (sigmoid(val_logits) > 0.5).astype(np.int64)
                    val_acc = (val_pred == y_val).mean()
                    if val_acc > best_val:
                        best_val = val_acc
                        best_cfg = (k, lr, l2, epochs)
                        best_params = params
    # Retrain on full data with best config
    k, lr, l2, epochs = best_cfg
    rng = np.random.default_rng(SEED)
    final_params = train_fm(X, y, X_val[:1], y_val[:1], k, lr, l2, epochs, rng)  # dummy val for interface
    bias, linear, emb = final_params
    # Pack weights: magic, d, k, C, bias, linear[d][256], emb[d][256][k]
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<5i", magic, d, k, C, 0)  # placeholder for bias index
    # We'll write bias as float, then linear, then emb
    buf = bytearray()
    buf += struct.pack("<f", bias)
    buf += linear.astype("<f4").tobytes()
    buf += emb.astype("<f4").tobytes()
    # Rewrite header with correct total size? Not needed; infer.c validates size.
    # Actually infer.c expects exact size; we'll compute total floats and pack header with total size?
    # Simpler: write header then data. infer.c will check nw == expected.
    # We'll write header first, then data. But we need to know total bytes for validation? infer.c computes expected from header.
    # So we write header with d,k,C and then the floats. infer.c will compute expected size from header.
    # Let's write header (magic, d, k, C) then bias, linear, emb.
    header = struct.pack("<4i", magic, d, k, C)
    data = struct.pack("<f", bias) + linear.astype("<f4").tobytes() + emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + data)

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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    // Expected size: 16 (header) + 4 (bias) + 4*d*256 (linear) + 4*d*256*k (emb)
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *ptr = (const float *)(hd + 4);
    float bias = *ptr++;
    const float *linear = ptr; ptr += (long)d * 256;
    const float *emb = ptr; // size d*256*k
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    // Temporary buffers
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_sq_emb = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        // Linear part
        float score = bias;
        for (int j = 0; j < d; j++) {
            score += linear[(long)j * 256 + xi[j]];
        }
        // Interaction part
        memset(sum_emb, 0, sizeof(float) * k);
        memset(sum_sq_emb, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *v = emb + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = v[u];
                sum_emb[u] += val;
                sum_sq_emb[u] += val * val;
            }
        }
        for (int u = 0; u < k; u++) {
            score += 0.5f * (sum_emb[u] * sum_emb[u] - sum_sq_emb[u]);
        }
        // Sigmoid and threshold
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_emb);
    free(sum_sq_emb);
    return 0;
}
```