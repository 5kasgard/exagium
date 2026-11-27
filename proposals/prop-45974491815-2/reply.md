```hypothesis
Family: label-emb (embedding + linear bottleneck). Learns a per-position byte embedding of size E, sums them, adds a bias, then applies a linear output layer. This factorizes the full one-hot weight matrix (rank E) reducing parameters and capturing character similarities. E is chosen on the tuning split (last 20% of train). If it fails to beat label-linear, the task likely requires full-rank interactions or non-linearities beyond a linear bottleneck.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed except E which is tuned)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_E = [8, 16, 32, 64, 128]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, E, C, rng, n_epochs):
    d = X.shape[1]
    # Parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    b_emb = np.zeros(E, np.float32)
    W_out = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = [emb, b_emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            # Sum embeddings: (batch, E)
            h = np.zeros((len(yb), E), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            h += b_emb
            logits = h @ W_out + b_out
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backward
            grad_W_out = h.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(0)
            grad_h = grad_logits @ W_out.T
            grad_b_emb = grad_h.sum(0)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_b_emb, grad_W_out, grad_b_out]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    d, E = params[0].shape[0], params[0].shape[2]
    h = np.zeros((X.shape[0], E), np.float32)
    for j in range(d):
        h += params[0][j, X[:, j]]
    h += params[1]
    logits = h @ params[2] + params[3]
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, len(X_train))))
    best_E, best_acc = None, -1.0
    for E in CANDIDATE_E:
        params = train_model(X_train, y_train, E, C, rng, n_epochs)
        acc = evaluate(X_val, y_val, params)
        if acc > best_acc:
            best_acc, best_E = acc, E
    # Retrain on full data with best_E
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, best_E, C, rng, n_epochs_full)
    emb, b_emb, W_out, b_out = final_params
    # Write weights.bin
    magic = 0x4C42454D  # "MEBL" in little-endian
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += b_emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C42454D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || E < 1 || E > 1024 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *b_emb = emb + (long)d * 256 * E;
    const float *W_out = b_emb + E;
    const float *b_out = W_out + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) h[e] = b_emb[e];
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e];
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int e = 0; e < E; e++) sum += h[e] * W_out[(long)e * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```