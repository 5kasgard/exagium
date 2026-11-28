```hypothesis
A two-hidden-layer MLP (embedding -> ReLU -> ReLU -> linear) with hyperparameters tuned on the last 20% of training data. The first layer is implemented as a precomputed per-byte lookup table (size 16×256×H1) so inference stays fast. By capturing pairwise and higher-order interactions between context bytes, it should exceed the linear model's 33.75% accuracy. If it fails to beat linear, the task is essentially linear or the extra capacity overfits.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1
# Grid search space (kept small to fit in time limit)
H1_OPTIONS = [64, 128]
H2_OPTIONS = [0, 64]          # 0 means single hidden layer
LR_OPTIONS = [1e-3, 3e-3]
L2_OPTIONS = [1e-4, 1e-3]
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

def encode_inputs(X, types):
    """Return list of encoding matrices E_j (256 x V_j) and total input dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def build_Z(X, enc):
    """Concatenate one-hot/ordinal encodings for all positions."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_model(Z, y, C, H1, H2, lr, l2, epochs, batch, rng):
    n, k = Z.shape
    # First layer weights: k x H1
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    if H2:
        W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
        b2 = np.zeros(H2, np.float32)
        W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
        b3 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2, W3, b3]
    else:
        W2 = (rng.uniform(-1, 1, (H1, C)) * math.sqrt(6.0 / H1)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            if H2:
                a2 = h1 @ W2 + b2
                h2 = np.maximum(a2, 0)
                logits = h2 @ W3 + b3
            else:
                logits = h1 @ W2 + b2
            g = softmax_grad(logits, yy)
            # Backward
            if H2:
                grad_W3 = h2.T @ g + l2 * W3
                grad_b3 = g.sum(0)
                gh2 = (g @ W3.T) * (a2 > 0)
                grad_W2 = h1.T @ gh2 + l2 * W2
                grad_b2 = gh2.sum(0)
                gh1 = (gh2 @ W2.T) * (a1 > 0)
                grad_W1 = z.T @ gh1 + l2 * W1
                grad_b1 = gh1.sum(0)
                grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            else:
                grad_W2 = h1.T @ g + l2 * W2
                grad_b2 = g.sum(0)
                gh1 = (g @ W2.T) * (a1 > 0)
                grad_W1 = z.T @ gh1 + l2 * W1
                grad_b1 = gh1.sum(0)
                grads = [grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(Z, y, C, params, H2):
    n = Z.shape[0]
    W1, b1 = params[0], params[1]
    a1 = Z @ W1 + b1
    h1 = np.maximum(a1, 0)
    if H2:
        W2, b2, W3, b3 = params[2], params[3], params[4], params[5]
        a2 = h1 @ W2 + b2
        h2 = np.maximum(a2, 0)
        logits = h2 @ W3 + b3
    else:
        W2, b2 = params[2], params[3]
        logits = h1 @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]

    # Encode once
    enc = encode_inputs(X, types)
    Z = build_Z(X, enc)

    # Split: last 20% for validation (tuning)
    n = X.shape[0]
    split = int(0.8 * n)
    Z_tr, Z_val = Z[:split], Z[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None

    # Grid search
    for H1 in H1_OPTIONS:
        for H2 in H2_OPTIONS:
            for lr in LR_OPTIONS:
                for l2 in L2_OPTIONS:
                    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
                    params = train_model(Z_tr, y_tr, C, H1, H2, lr, l2, epochs, BATCH, rng)
                    acc = evaluate(Z_val, y_val, C, params, H2)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (H1, H2, lr, l2, epochs)
                        best_params = params
    print(f"Best config: H1={best_cfg[0]}, H2={best_cfg[1]}, lr={best_cfg[2]}, l2={best_cfg[3]}, epochs={best_cfg[4]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    H1, H2, lr, l2, epochs = best_cfg
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_full = np.random.default_rng(SEED)  # fresh seed for final training
    final_params = train_model(Z, y, C, H1, H2, lr, l2, epochs, BATCH, rng_full)

    # Precompute lookup tables T for each position (256 x H1)
    T_list = []
    off = 0
    for E in enc:
        V = E.shape[1]
        T_list.append((E @ final_params[0][off:off+V]).astype(np.float32))
        off += V
    T = np.stack(T_list)  # shape (d, 256, H1)

    # Pack weights
    H1_val = H1
    H2_val = H2
    magic = 0x4D4C5032  # "MLP2" in little-endian
    header = struct.pack("<5i", magic, d, H1_val, H2_val, C)
    # Tables T
    tables_bytes = T.astype("<f4").tobytes()
    # Biases and weights
    b1 = final_params[1].astype("<f4").tobytes()
    if H2_val:
        W2 = final_params[2].astype("<f4").tobytes()
        b2 = final_params[3].astype("<f4").tobytes()
        W3 = final_params[4].astype("<f4").tobytes()
        b3 = final_params[5].astype("<f4").tobytes()
        body = tables_bytes + b1 + W2 + b2 + W3 + b3
    else:
        W2 = final_params[2].astype("<f4").tobytes()
        b2 = final_params[3].astype("<f4").tobytes()
        body = tables_bytes + b1 + W2 + b2

    open(sys.argv[4], "wb").write(header + body)

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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header values\n"); return 3; }

    // Expected file size
    long expected = 20
        + 4L * d * 256 * H1          // tables T
        + 4L * H1                    // b1
        + (H2 ?
            4L * H1 * H2 + 4L * H2   // W2, b2
            + 4L * H2 * C + 4L * C   // W3, b3
          : 4L * H1 * C + 4L * C);   // W2, b2 (output layer)
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (H2 ? (long)H1 * H2 : (long)H1 * C);
    const float *W3 = H2 ? b2 + H2 : NULL;
    const float *b3 = H2 ? W3 + (long)H2 * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = H2 ? malloc(sizeof(float) * H2) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // First hidden layer: accumulate table lookups
        for (int h = 0; h < H1; h++) acc1[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        // ReLU
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        if (H2) {
            // Second hidden layer
            for (int h = 0; h < H2; h++) acc2[h] = b2[h];
            for (int u = 0; u < H1; u++) {
                float val = acc1[u];
                if (val == 0.0f) continue;
                const float *w = W2 + (long)u * H2;
                for (int h = 0; h < H2; h++) acc2[h] += val * w[h];
            }
            for (int h = 0; h < H2; h++) if (acc2[h] < 0.0f) acc2[h] = 0.0f;
            // Output layer
            for (int c = 0; c < C; c++) scores[c] = b3[c];
            for (int u = 0; u < H2; u++) {
                float val = acc2[u];
                if (val == 0.0f) continue;
                const float *w = W3 + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += val * w[c];
            }
        } else {
            // Direct output layer
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int u = 0; u < H1; u++) {
                float val = acc1[u];
                if (val == 0.0f) continue;
                const float *w = W2 + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += val * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```