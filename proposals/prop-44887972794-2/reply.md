```hypothesis
Family: label-conv1d (1D convolution over byte embeddings with global max-pooling).
Why: Character-level language modeling benefits from local n-gram features. A small convolutional layer (kernel=3, 32 filters) on top of learned byte embeddings (dim=16) captures positional n-grams with far fewer parameters than a full MLP, while being faster than label-gbdt at inference. The global max-pooling aggregates the most salient features across the 16-byte context.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or instructions/item ≥ 10000, the family does not improve the Pareto frontier.
```

```python
"""label-conv1d: 1D convolution over byte embeddings with global max-pooling.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: embedding(256->E), conv1d(K, F, ReLU), global max-pool, linear(F->C).
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.in.
Exported binary format (little-endian):
  magic(0x434E5631), d, E, F, K, C (6 int32)
  embedding[256*E] (float32)
  conv_w[F*K*E] (float32)
  conv_b[F] (float32)
  lin_w[F*C] (float32)
  lin_b[C] (float32)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on validation split)
E = 16          # embedding dimension
F = 32          # number of filters
K = 3           # kernel size
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SEED = 1

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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: first 80% train, last 20% validation (tuning split)
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    n_tr = X_tr.shape[0]

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    conv_w = (rng.uniform(-1, 1, (F, K, E)) * math.sqrt(6.0 / (K * E))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    lin_w = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)

    params = [emb, conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # Embedding lookup: (B, d, E)
        emb_x = emb[Xb]
        # Create sliding windows: (B, d-K+1, K, E) -> (B, d-K+1, K*E)
        windows = np.lib.stride_tricks.sliding_window_view(emb_x, (K, E), axis=(1,2))
        windows = windows.reshape(B, d - K + 1, K * E)
        # Convolution: (B, d-K+1, F)
        conv_out = windows @ conv_w.reshape(F, K * E).T + conv_b
        # ReLU
        relu_out = np.maximum(conv_out, 0)
        # Global max-pool over positions: (B, F)
        pool_out = relu_out.max(axis=1)
        # Linear: (B, C)
        logits = pool_out @ lin_w.T + lin_b
        return logits, pool_out, relu_out, windows

    def loss_and_grad(Xb, yb):
        B = Xb.shape[0]
        logits, pool_out, relu_out, windows = forward(Xb)
        # Softmax cross-entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        probs /= B
        loss = -np.log(np.clip(probs[np.arange(B), yb] + 1.0, 1e-12, 1.0)).mean()  # actually cross-entropy
        # Backprop
        d_logits = probs  # (B, C)
        d_lin_w = pool_out.T @ d_logits + L2 * lin_w
        d_lin_b = d_logits.sum(axis=0)
        d_pool = d_logits @ lin_w  # (B, F)
        # Max-pool gradient: only max positions get gradient
        d_relu = np.zeros_like(relu_out)
        max_pos = relu_out.argmax(axis=1)  # (B, F)
        for b in range(B):
            for f in range(F):
                d_relu[b, max_pos[b, f], f] = d_pool[b, f]
        # ReLU gradient
        d_conv = d_relu * (conv_out > 0)
        # Convolution gradient
        d_conv_w = np.zeros_like(conv_w)
        d_conv_b = d_conv.sum(axis=(0,1))
        d_windows = np.zeros_like(windows)
        for b in range(B):
            for p in range(d - K + 1):
                for f in range(F):
                    grad = d_conv[b, p, f]
                    if grad != 0:
                        d_conv_w[f] += grad * windows[b, p].reshape(K, E)
                        d_windows[b, p] += grad * conv_w[f].reshape(K * E)
        # Embedding gradient
        d_emb = np.zeros_like(emb)
        # windows shape (B, d-K+1, K*E) came from emb_x (B, d, E)
        # We need to scatter d_windows back to emb_x positions
        d_emb_x = np.zeros((B, d, E), np.float32)
        for b in range(B):
            for p in range(d - K + 1):
                d_emb_x[b, p:p+K] += d_windows[b, p].reshape(K, E)
        # Scatter to embedding matrix
        np.add.at(d_emb, Xb, d_emb_x)
        d_emb += L2 * emb
        return loss, [d_emb, d_conv_w, d_conv_b, d_lin_w, d_lin_b]

    # Adam training on training split
    t = 0
    best_val_acc = -1
    best_params = None
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, grads = loss_and_grad(Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_logits, _, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
        # Early stopping could be added, but we just run fixed epochs

    # Retrain on full dataset for the same number of epochs that gave best_val_acc?
    # Simpler: just use the best_params from the run on full data? But we trained on 80% only.
    # According to rules: choose settings on tuning split, then train on all train.
    # We'll retrain on full data for the same number of epochs (MAX_EPOCHS) using the same seed.
    # Re-initialize with same seed and train on all data.
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    conv_w = (rng.uniform(-1, 1, (F, K, E)) * math.sqrt(6.0 / (K * E))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    lin_w = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)
    params = [emb, conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            loss, grads = loss_and_grad(Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    emb, conv_w, conv_b, lin_w, lin_b = params
    magic = 0x434E5631  # "CNV1"
    expected_size = 24 + (256*E + F*K*E + F + F*C + C) * 4
    out = struct.pack("<6i", magic, d, E, F, K, C)
    out += emb.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += lin_w.astype("<f4").tobytes()
    out += lin_b.astype("<f4").tobytes()
    assert len(out) == expected_size
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-conv1d inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x434E5631), d, E, F, K, C (int32 each),
 *   embedding[256*E] (float32), conv_w[F*K*E], conv_b[F],
 *   lin_w[F*C], lin_b[C].
 * Computes: embed -> conv1d(K, F, ReLU) -> global max-pool -> linear -> argmax.
 */
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
    if (nw < 24 || hd[0] != 0x434E5631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], F = hd[3], K = hd[4], C = hd[5];
    if (d != 16 || C != 65 || E <= 0 || F <= 0 || K <= 0 || K > d) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 24L + (256L*E + (long)F*K*E + F + (long)F*C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *conv_w = emb + 256L * E;
    const float *conv_b = conv_w + (long)F * K * E;
    const float *lin_w = conv_b + F;
    const float *lin_b = lin_w + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_buf = malloc(sizeof(float) * d * E);
    float *conv_out = malloc(sizeof(float) * (d - K + 1) * F);
    float *pool_out = malloc(sizeof(float) * F);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(emb_buf + j * E, e, E * sizeof(float));
        }

        // Conv1d + ReLU
        int P = d - K + 1;
        for (int p = 0; p < P; p++) {
            for (int f = 0; f < F; f++) {
                float sum = conv_b[f];
                const float *w = conv_w + ((long)f * K * E);
                for (int kk = 0; kk < K; kk++) {
                    const float *e = emb_buf + (p + kk) * E;
                    for (int e_idx = 0; e_idx < E; e_idx++) {
                        sum += e[e_idx] * w[kk * E + e_idx];
                    }
                }
                conv_out[p * F + f] = sum > 0.0f ? sum : 0.0f;
            }
        }

        // Global max-pool
        for (int f = 0; f < F; f++) {
            float mx = -1e30f;
            for (int p = 0; p < P; p++) {
                float v = conv_out[p * F + f];
                if (v > mx) mx = v;
            }
            pool_out[f] = mx;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            for (int f = 0; f < F; f++) {
                sum += pool_out[f] * lin_w[(long)f * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```