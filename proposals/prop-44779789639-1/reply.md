```hypothesis
We propose "label-emb-mlp": learn an embedding vector (size E) for each of the 17 possible pixel intensities (0–16) at each of the 64 positions, concatenate them (64×E features), then apply a one-hidden-layer ReLU MLP (size H) and a linear output layer. This replaces the fixed z‑scoring of label‑linear with learned per‑intensity features, capturing non‑linear pixel responses while sharing parameters across positions. On the tuning split (last 20 % of train.in) we grid‑search E∈{4,8,16} and H∈{16,32,64}; if the best configuration does not beat label‑linear’s dev accuracy (0.9613) by ≥0.5 % the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters for training
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Grid for tuning split
E_CHOICES = [4, 8, 16]
H_CHOICES = [16, 32, 64]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, epochs, rng, verbose=False):
    n, _ = X.shape
    num_values = 17  # pixel intensities 0..16
    # Initialize parameters
    emb = (rng.uniform(-1, 1, (d, num_values, E)) * math.sqrt(6.0 / (num_values * E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Embedding lookup: (d, batch, E) -> (batch, d, E) -> (batch, d*E)
        x_emb = emb[np.arange(d), X_batch.T].transpose(1, 0, 2).reshape(batch, d * E)
        h = x_emb @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        return x_emb, h, h_relu, logits

    def loss_and_grad(X_batch, y_batch):
        batch = X_batch.shape[0]
        x_emb, h, h_relu, logits = forward(X_batch)
        # Softmax cross-entropy
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        probs /= batch
        # Gradients
        dW2 = h_relu.T @ probs + L2 * W2
        db2 = probs.sum(0)
        dh_relu = probs @ W2.T
        dh = dh_relu * (h > 0)
        dW1 = x_emb.T @ dh + L2 * W1
        db1 = dh.sum(0)
        dx_emb = dh @ W1.T  # (batch, d*E)
        # Scatter back to embeddings
        demb = np.zeros_like(emb)
        dx_emb_3d = dx_emb.reshape(batch, d, E)
        for j in range(d):
            vals = X_batch[:, j]
            np.add.at(demb[j], vals, dx_emb_3d[:, j, :])
        demb += L2 * emb
        return [demb, dW1, db1, dW2, db2]

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def accuracy(X, y, params, d, C, E, H):
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        X_batch, y_batch = X[s:s+BATCH], y[s:s+BATCH]
        batch = X_batch.shape[0]
        x_emb = emb[np.arange(d), X_batch.T].transpose(1, 0, 2).reshape(batch, d * E)
        h = x_emb @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == y_batch).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # Tuning split: last 20% in file order
    split = int(0.8 * n_all)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    best_acc = -1.0
    best_E, best_H = E_CHOICES[0], H_CHOICES[0]

    # Grid search on tuning split
    for E in E_CHOICES:
        for H in H_CHOICES:
            rng = np.random.default_rng(SEED + E * 100 + H)
            params = train_model(X_train, y_train, d, C, E, H, epochs, rng)
            acc = accuracy(X_tune, y_tune, params, d, C, E, H)
            if acc > best_acc:
                best_acc, best_E, best_H = acc, E, H

    # Retrain on full data with best (E, H)
    rng = np.random.default_rng(SEED + best_E * 100 + best_H)
    final_params = train_model(X_all, y_all, d, C, best_E, best_H, epochs, rng)
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    # Magic: 0x4C424D45 (little-endian 'E','M','B','L')
    magic = 0x4C424D45
    num_values = 17
    header = struct.pack("<6i", magic, d, num_values, best_E, best_H, C)
    # All arrays in little-endian float32
    out = header
    out += emb.astype("<f4").tobytes()
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
    if (nw < 24 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_values = hd[2], E = hd[3], H = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != 64 || num_values != 17) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 24L
        + 4L * d * num_values * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *W1 = emb + (long)d * num_values * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * num_values + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) x_emb[j * E + ee] = e[ee];
        }

        // Hidden layer: h = ReLU(x_emb @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += x_emb[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```