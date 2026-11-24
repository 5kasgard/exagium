```hypothesis
Family: label-mlp-emb (position-specific embeddings + 1 hidden ReLU layer).
Why: The linear model (label-linear) treats each position independently; adding a hidden layer allows modeling interactions between context positions. Using a separate embedding dimension (E=32) before expanding to hidden size (H=64) acts as a bottleneck that regularizes and reduces parameters vs. a full one-hot first layer. Tuning on the last 20% of train.in with early stopping should prevent overfitting and yield higher accuracy than label-linear (0.3375) at modest CPU cost (~2x label-linear).
Failure signal: dev accuracy ≤ 0.3375 or inference cost > 20k Ir/item.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on the tuning split)
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
E = 32          # embedding dimension per position
H = 64          # hidden layer size
MAX_EPOCHS = 200
PATIENCE = 10   # early stopping patience


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65

    # Load data
    X_list = records(sys.argv[2])
    y_list = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_list), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_list], dtype=np.int64)
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx, val_idx = idx[:split], idx[split:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: (d, 256, E)
    Emb = rng.normal(0, 1/math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    # Hidden layer: (E, H)
    W1 = rng.normal(0, math.sqrt(2.0/E), size=(E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = rng.normal(0, 1/math.sqrt(H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [Emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(X_batch):
        # X_batch: (N, d) uint8
        N = X_batch.shape[0]
        # Sum embeddings over positions
        S = np.zeros((N, E), np.float32)
        for j in range(d):
            S += Emb[j, X_batch[:, j], :]
        # Hidden layer
        H_pre = S @ W1 + b1
        H_act = np.maximum(H_pre, 0, dtype=np.float32)
        # Output
        logits = H_act @ W2 + b2
        return S, H_pre, H_act, logits

    def loss_and_grad(X_batch, y_batch):
        N = X_batch.shape[0]
        S, H_pre, H_act, logits = forward(X_batch)
        # Softmax cross-entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(N), y_batch] + 1e-12).mean()
        # Gradient on logits
        dlogits = probs
        dlogits[np.arange(N), y_batch] -= 1.0
        dlogits /= N
        # Output layer grads
        dW2 = H_act.T @ dlogits
        db2 = dlogits.sum(axis=0)
        # Hidden layer grads
        dH_act = dlogits @ W2.T
        dH_pre = dH_act * (H_pre > 0)
        dW1 = S.T @ dH_pre
        db1 = dH_pre.sum(axis=0)
        # Embedding grads
        dS = dH_pre @ W1.T
        dEmb = np.zeros_like(Emb)
        for j in range(d):
            # one-hot scatter for position j
            idx_j = X_batch[:, j]
            # Efficient accumulation via matrix multiply with one-hot (N,256)
            one_hot = np.zeros((N, 256), np.float32)
            one_hot[np.arange(N), idx_j] = 1.0
            dEmb[j] = one_hot.T @ dS
        # L2 regularization gradients
        dEmb += L2 * Emb
        dW1 += L2 * W1
        dW2 += L2 * W2
        return loss, [dEmb, dW1, db1, dW2, db2]

    def evaluate(X_eval, y_eval):
        _, _, _, logits = forward(X_eval)
        preds = logits.argmax(axis=1)
        return (preds == y_eval).mean()

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training data
        order = rng.permutation(train_idx)
        train_loss = 0.0
        train_batches = 0
        for s in range(0, len(train_idx), BATCH):
            batch_idx = order[s:s+BATCH]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            loss, grads = loss_and_grad(X_batch, y_batch)
            train_loss += loss
            train_batches += 1
            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        val_acc = evaluate(X[val_idx], y[val_idx])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Retrain on full data for best_epoch epochs
    if best_params is not None:
        params = best_params
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for epoch in range(1, best_epoch + 1):
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                batch_idx = order[s:s+BATCH]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                _, grads = loss_and_grad(X_batch, y_batch)
                t += 1
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                    m_hat = m[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    Emb, W1, b1, W2, b2 = params

    # Write weights.bin
    # Magic: 0x4D4C5045 ('EM' in ascii? but we need a unique int)
    magic = 0x4D4C5045  # 'EM' 'P' 'E'? Actually we can use 'EMLP' -> 0x504C4D45? Let's use 0x454D4C50 (little-endian 'P','L','M','E')
    # We'll use 0x454D4C50 as magic (ASCII 'P','L','M','E' in little-endian order? Actually struct.pack('<I', 0x454D4C50) gives bytes P L M E)
    # But we just need a unique identifier; we'll use 0x454D4C50.
    header = struct.pack("<5i", 0x454D4C50, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = Emb.astype("<f4").tobytes()
    # W1: E * H
    w1_bytes = W1.astype("<f4").tobytes()
    # b1: H
    b1_bytes = b1.astype("<f4").tobytes()
    # W2: H * C
    w2_bytes = W2.astype("<f4").tobytes()
    # b2: C
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + w1_bytes + b1_bytes + w2_bytes + b2_bytes
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4C50) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dimensions\n"); return 3; }
    // Expected size: header(20) + d*256*E*4 + E*H*4 + H*4 + H*C*4 + C*4
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *S = (float *)malloc(sizeof(float) * E);
    float *H_pre = (float *)malloc(sizeof(float) * H);
    float *H_act = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        memset(S, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *emb = Emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) S[e] += emb[e];
        }

        // Hidden layer: H_pre = S @ W1 + b1
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += S[e] * W1[(long)e * H + h];
            H_pre[h] = sum;
            H_act[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = H_act @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += H_act[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(S); free(H_pre); free(H_act); free(logits);
    return 0;
}
```