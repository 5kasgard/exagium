```hypothesis
Family: label-conv1d - a tiny 1D CNN with kernel size 3, 16 filters, global average pooling, and linear classifier.
Why: The 8x8 digit images have strong local spatial structure; weight-sharing convolutions exploit translation invariance and drastically reduce parameters vs. MLP, while capturing local patterns (strokes, curves) that linear models miss. Fixed kernel size 3 and 16 filters keep inference fast (~3k FLOPs/item) and weights tiny (<2KB).
Failure signal: If dev accuracy <= 0.9613 (label-linear) or Ir/item > 10000, the family does not justify its complexity.
```
```python
"""label-conv1d: tiny 1D CNN for 8x8 optdigits (64 ordinal pixels).
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: per-pixel z-score -> Conv1D(16 filters, kernel=3, pad=1) -> ReLU -> GlobalAvgPool -> Linear(10).
Training: Adam, L2, tuned on last 20% of train.in (file order), then retrained on all.
Export: magic 0x434E5631, d=64, C=10, K=3, F=16; float32 mu[64], sd[64], conv_W[F][K], conv_b[F], fc_W[F][C], fc_b[C]."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
K, F, C = 3, 16, 10
SEED = 1

# Tuning grid (eval on validation split)
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
TUNE_EPOCHS = 15
FINAL_EPOCHS = 50
BATCH = 64
SAMPLES = 600000  # max samples per epoch decision


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """logits: (B, C), y: (B,) -> grad wrt logits: (B, C)"""
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B


def evaluate(X, y, mu, sd, conv_W, conv_b, fc_W, fc_b):
    """Return accuracy on (X, y). X is uint8 (n, 64)."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s + BATCH].astype(np.float32)
        yb = y[s:s + BATCH]
        xb = (xb - mu) / sd
        B = xb.shape[0]
        # Conv + ReLU
        out = np.zeros((B, F, 64), dtype=np.float32)
        for f in range(F):
            for p in range(64):
                acc = conv_b[f]
                for k in range(K):
                    idx = p + k - 1
                    if 0 <= idx < 64:
                        acc += xb[:, idx] * conv_W[f, k]
                out[:, f, p] = np.maximum(acc, 0)
        # Global average pool
        pool = out.mean(axis=2)
        # Linear
        logits = pool @ fc_W + fc_b
        pred = logits.argmax(axis=1)
        correct += np.sum(pred == yb)
    return correct / n


def train_loop(X, y, mu, sd, conv_W, conv_b, fc_W, fc_b, lr, l2, epochs, rng):
    """In-place training for given epochs. Returns final weights."""
    n = X.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    max_steps = min(epochs * steps_per_epoch, math.ceil(SAMPLES / n) * steps_per_epoch)
    # Adam state
    params = [conv_W, conv_b, fc_W, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx].astype(np.float32)
            yb = y[idx]
            xb = (xb - mu) / sd
            B = xb.shape[0]
            # Forward
            out = np.zeros((B, F, 64), dtype=np.float32)
            for f in range(F):
                for p in range(64):
                    acc = conv_b[f]
                    for k in range(K):
                        idx_k = p + k - 1
                        if 0 <= idx_k < 64:
                            acc += xb[:, idx_k] * conv_W[f, k]
                    out[:, f, p] = np.maximum(acc, 0)
            pool = out.mean(axis=2)
            logits = pool @ fc_W + fc_b
            # Backward
            dlogits = softmax_cross_entropy_grad(logits, yb)
            dfc_W = pool.T @ dlogits + l2 * fc_W
            dfc_b = dlogits.sum(axis=0)
            dpool = dlogits @ fc_W.T
            dout = dpool[:, :, np.newaxis] / 64.0
            dout[out <= 0] = 0
            dconv_W = np.zeros_like(conv_W)
            dconv_b = np.zeros_like(conv_b)
            for f in range(F):
                for p in range(64):
                    for k in range(K):
                        idx_k = p + k - 1
                        if 0 <= idx_k < 64:
                            dconv_W[f, k] += np.sum(dout[:, f, p] * xb[:, idx_k])
                dconv_b[f] = np.sum(dout[:, f, :])
            dconv_W += l2 * conv_W
            grads = [dconv_W, dconv_b, dfc_W, dfc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return conv_W, conv_b, fc_W, fc_b


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64 and card["classes"] == 10
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    # Compute normalization on training split only
    mu = X[train_idx].mean(axis=0).astype(np.float32)
    sd = X[train_idx].std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    rng = np.random.default_rng(SEED)

    # Hyperparameter tuning on validation split
    best_acc = -1.0
    best_cfg = None
    best_weights = None
    for lr in LR_GRID:
        for l2 in L2_GRID:
            # Initialize
            conv_W = (rng.uniform(-1, 1, (F, K)) * math.sqrt(6.0 / K)).astype(np.float32)
            conv_b = np.zeros(F, np.float32)
            fc_W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
            fc_b = np.zeros(C, np.float32)
            # Train on train split
            conv_W, conv_b, fc_W, fc_b = train_loop(
                X[train_idx], y[train_idx], mu, sd,
                conv_W, conv_b, fc_W, fc_b, lr, l2, TUNE_EPOCHS, rng
            )
            acc = evaluate(X[val_idx], y[val_idx], mu, sd, conv_W, conv_b, fc_W, fc_b)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (lr, l2)
                best_weights = (conv_W.copy(), conv_b.copy(), fc_W.copy(), fc_b.copy())
                print(f"Tune: lr={lr}, l2={l2}, val_acc={acc:.4f}", file=sys.stderr)

    print(f"Best: lr={best_cfg[0]}, l2={best_cfg[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparams
    conv_W, conv_b, fc_W, fc_b = best_weights
    conv_W, conv_b, fc_W, fc_b = train_loop(
        X, y, mu, sd, conv_W, conv_b, fc_W, fc_b,
        best_cfg[0], best_cfg[1], FINAL_EPOCHS, rng
    )

    # Export weights
    # Magic: 0x434E5631 ('CNV1' little-endian)
    # Order: d, C, K, F, mu[64], sd[64], conv_W[F][K], conv_b[F], fc_W[F][C], fc_b[C]
    hdr = struct.pack("<5i", 0x434E5631, 64, C, K, F)
    body = mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    body += conv_W.astype("<f4").tobytes() + conv_b.astype("<f4").tobytes()
    body += fc_W.astype("<f4").tobytes() + fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* label-conv1d inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 pixels 0..16). Output: 1-byte class (0..9).
 * Weights format (little-endian):
 *   int32 magic=0x434E5631, d=64, C=10, K=3, F=16
 *   float32 mu[64], sd[64]
 *   float32 conv_W[F][K], conv_b[F], fc_W[F][C], fc_b[C] */
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
    if (nw < 20 || hd[0] != 0x434E5631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3], F = hd[4];
    if (d != 64 || C != 10 || K != 3 || F != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * (64 + 64 + F * K + F + F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 5);
    const float *sd = mu + 64;
    const float *conv_W = sd + 64;
    const float *conv_b = conv_W + F * K;
    const float *fc_W = conv_b + F;
    const float *fc_b = fc_W + F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float xnorm[64];
    float conv_out[16][64];
    float pool[16];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize
        for (int j = 0; j < 64; j++) xnorm[j] = (xi[j] - mu[j]) / sd[j];

        // Conv1D + ReLU
        for (int f = 0; f < F; f++) {
            for (int p = 0; p < 64; p++) {
                float acc = conv_b[f];
                for (int kk = 0; kk < K; kk++) {
                    int idx = p + kk - 1;
                    if (idx >= 0 && idx < 64) acc += xnorm[idx] * conv_W[f * K + kk];
                }
                conv_out[f][p] = acc > 0.0f ? acc : 0.0f;
            }
        }

        // Global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int p = 0; p < 64; p++) sum += conv_out[f][p];
            pool[f] = sum / 64.0f;
        }

        // Linear classifier
        for (int c = 0; c < C; c++) {
            float acc = fc_b[c];
            for (int f = 0; f < F; f++) acc += pool[f] * fc_W[f * C + c];
            logits[c] = acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```