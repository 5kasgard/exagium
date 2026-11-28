```hypothesis
Propose "label-emb": position-specific byte embeddings (size D) summed and linearly projected to 65 classes.
This low-rank factorization of the linear weight matrix captures pairwise interactions between positions
while keeping inference fast (O(d*D + D*C) ops). It should beat label-linear (0.3375) on dev accuracy
at similar or lower CPU cost. Failure: dev accuracy ≤ 0.3375 or cost per item not lower than label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
D_CHOICES = [8, 16, 32, 64]
LR_CHOICES = [0.01, 0.03, 0.1]
EPOCHS_CHOICES = [10, 20, 30]
BATCH = 64
SEED = 1
L2 = 1e-4
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (batch, C), y: (batch,) int
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)  # gradient w.r.t logits

def train_eval(X_tr, y_tr, X_val, y_val, D, lr, epochs, rng):
    n_tr, d = X_tr.shape
    C = 65
    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W_out = (rng.normal(0, 0.01, (D, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # Forward: sum embeddings
            summed = np.zeros((len(xb), D), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j]]
            logits = summed @ W_out + b_out
            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # Gradients
            g_summed = g_logits @ W_out.T
            g_W = summed.T @ g_logits + L2 * W_out
            g_b = g_logits.sum(axis=0)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_summed)
            g_emb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [(emb, g_emb, m_emb, v_emb),
                                      (W_out, g_W, m_W, v_W),
                                      (b_out, g_b, m_b, v_b)]:
                m *= BETA1; m += (1 - BETA1) * grad
                v *= BETA2; v += (1 - BETA2) * grad * grad
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + EPS)
    # Validation accuracy
    summed_val = np.zeros((len(X_val), D), np.float32)
    for j in range(d):
        summed_val += emb[j, X_val[:, j]]
    logits_val = summed_val @ W_out + b_out
    pred = logits_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, emb, W_out, b_out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = len(X)
    split = int(0.8 * n)  # last 20% for tuning
    X_tr_full, y_tr_full = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCHS_CHOICES:
                acc, emb, W_out, b_out = train_eval(X_tr_full, y_tr_full, X_val, y_val, D, lr, epochs, rng)
                if acc > best_acc:
                    best_acc, best_cfg = acc, (D, lr, epochs)
                    best_params = (emb.copy(), W_out.copy(), b_out.copy())
    print(f"Best tuning acc: {best_acc:.4f} with D={best_cfg[0]}, lr={best_cfg[1]}, epochs={best_cfg[2]}", file=sys.stderr)

    # Retrain on full training set with best config
    D, lr, epochs = best_cfg
    rng_full = np.random.default_rng(SEED)  # fresh deterministic seed
    _, emb, W_out, b_out = train_eval(X, y, X_val, y_val, D, lr, epochs, rng_full)

    # Write weights.bin: magic, d, D, C; then embeddings[d][256][D], W_out[D][C], b_out[C] all float32 little-endian
    magic = 0x4C424D45  # "L B M E"
    header = struct.pack("<4i", magic, d, D, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W_out.astype("<f4").tobytes()
    b_bytes = b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

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
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x4C424D45) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int D = (int32_t)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    int C = (int32_t)(wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24);
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights: invalid dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wb + 16);
    const float *W_out = emb + (long)d * 256 * D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < D; u++) summed[u] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) summed[u] += e[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int u = 0; u < D; u++) {
            float su = summed[u];
            if (su != 0.0f) {
                const float *w = W_out + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += su * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```