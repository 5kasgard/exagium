```hypothesis
Family: label-tinycnn — a tiny 1D CNN over learned byte embeddings.
Why: Character-level language modeling benefits from local n-gram patterns; a depth-1 conv with max-pooling captures these
more parameter-efficiently than a linear one-hot model. With ~12k parameters it should exceed label-linear's 33.8%
accuracy while keeping inference under 10k Ir/item.
Failure: If dev accuracy ≤ 0.3375 or Ir/item > 10000, the family does not justify its complexity.
```
```python
"""label-tinycnn: 1D CNN over byte embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only.
Architecture: Embed(256→E) → Conv1d(E→F, kernel=K, ReLU) → GlobalMaxPool → (FC(H, ReLU)?) → FC(C).
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data."""
import json, math, struct, sys
import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 30
# Hyperparameter grid for tuning split
GRID = [
    {"E": 8, "F": 16, "K": 3, "H": 0},
    {"E": 8, "F": 32, "K": 3, "H": 0},
    {"E": 16, "F": 16, "K": 3, "H": 0},
    {"E": 16, "F": 32, "K": 3, "H": 64},
    {"E": 16, "F": 32, "K": 4, "H": 64},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(cfg, C, rng):
    E, F, K, H = cfg["E"], cfg["F"], cfg["K"], cfg["H"]
    # Embedding: 256 x E
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Conv: F x E x K
    fan_in = E * K
    conv_w = (rng.uniform(-1, 1, (F, E, K)) * math.sqrt(6.0 / fan_in)).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    params = [emb, conv_w, conv_b]
    if H:
        fc1_w = (rng.uniform(-1, 1, (H, F)) * math.sqrt(6.0 / F)).astype(np.float32)
        fc1_b = np.zeros(H, np.float32)
        fc2_w = (rng.uniform(-1, 1, (C, H)) * math.sqrt(6.0 / H)).astype(np.float32)
        fc2_b = np.zeros(C, np.float32)
        params += [fc1_w, fc1_b, fc2_w, fc2_b]
    else:
        fc_w = (rng.uniform(-1, 1, (C, F)) * math.sqrt(6.0 / F)).astype(np.float32)
        fc_b = np.zeros(C, np.float32)
        params += [fc_w, fc_b]
    return params

def forward(params, X, cfg, train=True):
    """X: (N, 16) uint8. Returns logits (N, C)."""
    E, F, K, H = cfg["E"], cfg["F"], cfg["K"], cfg["H"]
    emb, conv_w, conv_b = params[0], params[1], params[2]
    N = X.shape[0]
    # Embedding lookup: (N, 16, E)
    x_emb = emb[X]  # (N, 16, E)
    # Conv1d: valid padding, stride 1
    P = 16 - K + 1  # number of positions
    # im2col: (N, P, E*K)
    col = np.lib.stride_tricks.sliding_window_view(x_emb, (K, E), axis=(1,2))  # (N, P, K, E)
    col = col.reshape(N, P, K * E)
    # Conv weights: (F, E*K)
    conv_w_flat = conv_w.reshape(F, -1)
    # (N, P, F)
    conv_out = col @ conv_w_flat.T + conv_b  # (N, P, F)
    conv_out = np.maximum(conv_out, 0)  # ReLU
    # Global max pool over positions: (N, F)
    pooled = np.max(conv_out, axis=1)
    if H:
        fc1_w, fc1_b, fc2_w, fc2_b = params[3], params[4], params[5], params[6]
        hidden = pooled @ fc1_w.T + fc1_b  # (N, H)
        hidden = np.maximum(hidden, 0)
        logits = hidden @ fc2_w.T + fc2_b
    else:
        fc_w, fc_b = params[3], params[4]
        logits = pooled @ fc_w.T + fc_b
    return logits

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of loss w.r.t logits, shape (N, C)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

def train_model(X, y, X_val, y_val, cfg, C, rng):
    params = init_params(cfg, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_val, best_params = -1, None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits = forward(params, xb, cfg)
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # Backprop
            E, F, K, H = cfg["E"], cfg["F"], cfg["K"], cfg["H"]
            emb, conv_w, conv_b = params[0], params[1], params[2]
            N_b = xb.shape[0]
            P = 16 - K + 1
            # Forward pass intermediates for backprop
            x_emb = emb[xb]  # (N, 16, E)
            col = np.lib.stride_tricks.sliding_window_view(x_emb, (K, E), axis=(1,2)).reshape(N_b, P, K*E)
            conv_pre = col @ conv_w.reshape(F, -1).T + conv_b
            conv_out = np.maximum(conv_pre, 0)
            pool_idx = np.argmax(conv_out, axis=1)  # (N, F)
            pooled = conv_out[np.arange(N_b)[:,None], pool_idx, np.arange(F)]
            if H:
                fc1_w, fc1_b, fc2_w, fc2_b = params[3], params[4], params[5], params[6]
                hidden_pre = pooled @ fc1_w.T + fc1_b
                hidden = np.maximum(hidden_pre, 0)
                # grads
                g_fc2_w = g.T @ hidden + L2 * fc2_w
                g_fc2_b = g.sum(0)
                g_hidden = g @ fc2_w
                g_hidden_pre = g_hidden * (hidden_pre > 0)
                g_fc1_w = g_hidden_pre.T @ pooled + L2 * fc1_w
                g_fc1_b = g_hidden_pre.sum(0)
                g_pooled = g_hidden_pre @ fc1_w
            else:
                fc_w, fc_b = params[3], params[4]
                g_fc_w = g.T @ pooled + L2 * fc_w
                g_fc_b = g.sum(0)
                g_pooled = g @ fc_w
            # Backprop through max pool
            g_conv_out = np.zeros_like(conv_out)
            g_conv_out[np.arange(N_b)[:,None], pool_idx, np.arange(F)] = g_pooled
            # Backprop through ReLU
            g_conv_pre = g_conv_out * (conv_pre > 0)
            # Backprop through conv
            g_conv_w = g_conv_pre.reshape(-1, F).T @ col.reshape(-1, K*E) + L2 * conv_w.reshape(F, -1)
            g_conv_b = g_conv_pre.sum((0,1))
            g_col = g_conv_pre @ conv_w.reshape(F, -1)  # (N, P, E*K)
            g_col = g_col.reshape(N_b, P, K, E)
            # Backprop through embedding
            g_emb = np.zeros_like(emb)
            # x_emb[i, j, :] gets gradient from g_col[i, j-k+1, k-1, :] for valid k
            # Use scatter add via loop (small sizes)
            for i in range(N_b):
                for p in range(P):
                    for k in range(K):
                        byte_val = xb[i, p + k]
                        g_emb[byte_val] += g_col[i, p, k]
            grads = [g_emb, g_conv_w.reshape(F, E, K), g_conv_b]
            if H:
                grads += [g_fc1_w, g_fc1_b, g_fc2_w, g_fc2_b]
            else:
                grads += [g_fc_w, g_fc_b]
            # Adam update
            t = epoch * (n // BATCH) + s // BATCH + 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation check
        val_logits = forward(params, X_val, cfg, train=False)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_params

def pack_weights(params, cfg, d, C):
    """Pack weights into binary format for infer.c.
    Magic: 0x434E4E31 ('CNN1'), then d, E, F, K, H, C (6 int32).
    Then float32 arrays in order: emb(256,E), conv_w(F,E,K), conv_b(F),
    [fc1_w(H,F), fc1_b(H), fc2_w(C,H), fc2_b(C)] or [fc_w(C,F), fc_b(C)]."""
    E, F, K, H = cfg["E"], cfg["F"], cfg["K"], cfg["H"]
    magic = 0x434E4E31
    header = struct.pack("<7i", magic, d, E, F, K, H, C)
    emb, conv_w, conv_b = params[0], params[1], params[2]
    body = emb.astype("<f4").tobytes()
    body += conv_w.astype("<f4").tobytes()
    body += conv_b.astype("<f4").tobytes()
    if H:
        fc1_w, fc1_b, fc2_w, fc2_b = params[3], params[4], params[5], params[6]
        body += fc1_w.astype("<f4").tobytes()
        body += fc1_b.astype("<f4").tobytes()
        body += fc2_w.astype("<f4").tobytes()
        body += fc2_b.astype("<f4").tobytes()
    else:
        fc_w, fc_b = params[3], params[4]
        body += fc_w.astype("<f4").tobytes()
        body += fc_b.astype("<f4").tobytes()
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    # Split: last 20% for tuning
    n = X_all.shape[0]
    split = n - n // 5
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    # Grid search on validation
    best_cfg, best_val_acc, best_params = None, -1, None
    for cfg in GRID:
        params = train_model(X_train, y_train, X_val, y_val, cfg, C, rng)
        val_logits = forward(params, X_val, cfg, train=False)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = cfg
            best_params = params
    # Retrain on full data with best config
    rng_full = np.random.default_rng(SEED)
    final_params = train_model(X_all, y_all, X_all[:1], y_all[:1], best_cfg, C, rng_full)  # dummy val
    # Pack and write
    out = pack_weights(final_params, best_cfg, d, C)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tinycnn inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64). */
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
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
    if (nw < 28 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], F = hd[3], K = hd[4], H = hd[5], C = hd[6];
    if (d != 16 || C < 1 || C > 256 || E < 1 || F < 1 || K < 1 || K > d) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    int P = d - K + 1;
    long expected = 28L
        + (long)256 * E
        + (long)F * E * K + F
        + (H ? (long)H * F + H + (long)C * H + C : (long)C * F + C);
    if (nw != 4 + expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 7);
    const float *emb = p; p += 256 * E;
    const float *conv_w = p; p += (long)F * E * K;
    const float *conv_b = p; p += F;
    const float *fc1_w = H ? p : NULL; if (H) p += (long)H * F;
    const float *fc1_b = H ? p : NULL; if (H) p += H;
    const float *fc2_w = H ? p : p; if (H) p += (long)C * H; else p += (long)C * F;
    const float *fc2_b = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *conv_out = malloc(sizeof(float) * P * F);
    float *pooled = malloc(sizeof(float) * F);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(x_emb + (long)j * E, e, sizeof(float) * E);
        }
        // Conv1d + ReLU
        for (int pos_idx = 0; pos_idx < P; pos_idx++) {
            for (int f = 0; f < F; f++) {
                float sum = conv_b[f];
                for (int kk = 0; kk < K; kk++) {
                    const float *w = conv_w + ((long)f * E * K + kk * E);
                    const float *in = x_emb + (long)(pos_idx + kk) * E;
                    for (int e = 0; e < E; e++) sum += in[e] * w[e];
                }
                conv_out[(long)pos_idx * F + f] = sum > 0 ? sum : 0;
            }
        }
        // Global max pool
        for (int f = 0; f < F; f++) {
            float m = conv_out[f];
            for (int pos_idx = 1; pos_idx < P; pos_idx++) {
                float v = conv_out[(long)pos_idx * F + f];
                if (v > m) m = v;
            }
            pooled[f] = m;
        }
        // FC layers
        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = fc1_b[h];
                for (int f = 0; f < F; f++) sum += pooled[f] * fc1_w[(long)h * F + f];
                hidden[h] = sum > 0 ? sum : 0;
            }
            for (int c = 0; c < C; c++) {
                float sum = fc2_b[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * fc2_w[(long)c * H + h];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = fc2_b[c];
                for (int f = 0; f < F; f++) sum += pooled[f] * fc2_w[(long)c * F + f];
                logits[c] = sum;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```