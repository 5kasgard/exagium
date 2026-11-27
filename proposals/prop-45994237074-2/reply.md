```hypothesis
Family: label-embed-mlp (shared character embeddings + 1-hidden-layer MLP).
Why: The task is character-level language modeling; sharing embeddings across the 16 context positions captures character similarities while keeping parameters low. A small ReLU layer on the concatenated embeddings adds non-linearity missing in label-linear. This should exceed label-linear's 0.3375 accuracy with far fewer FLOPs than label-gbdt.
Failure signal: Dev accuracy ≤ 0.3375 or inference cost > label-linear (≈7k Ir/item).
```
```python
"""label-embed-mlp: shared embeddings + 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic(0x454D4245), d, H, C, context_len; then float32 arrays:
  emb[256][d], W1[H][context_len*d], b1[H], W2[C][H], b2[C].
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train)
D_EMB = 16          # embedding dimension
H = 64              # hidden units
SEED = 1
LR = 1e-3
L2 = 1e-4
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

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t. logits (already averaged over batch)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    ctx_len = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, ctx_len)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    emb = (rng.uniform(-1, 1, (256, D_EMB)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (H, ctx_len * D_EMB)) * math.sqrt(6.0 / (ctx_len * D_EMB))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (C, H)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, ctx_len)
            yb = y[idx]          # (B,)

            # Forward
            # Embedding lookup: (B, ctx_len, D_EMB)
            x_emb = emb[xb]                    # (B, ctx_len, D_EMB)
            x_flat = x_emb.reshape(len(yb), -1)  # (B, ctx_len*D_EMB)
            h_pre = x_flat @ W1.T + b1         # (B, H)
            h = np.maximum(h_pre, 0)           # ReLU
            logits = h @ W2.T + b2             # (B, C)

            # Backward
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)  # (B, C)
            # grad W2, b2
            gW2 = g.T @ h + L2 * W2
            gb2 = g.sum(axis=0)
            # grad hidden
            gh = g @ W2                        # (B, H)
            gh[h_pre <= 0] = 0                 # ReLU mask
            # grad W1, b1
            gW1 = gh.T @ x_flat + L2 * W1
            gb1 = gh.sum(axis=0)
            # grad embeddings
            g_emb = np.zeros_like(emb)
            gh_reshaped = gh.reshape(len(yb), ctx_len, D_EMB) @ W1.reshape(ctx_len, D_EMB, H).transpose(1, 0, 2)
            # gh_reshaped is (B, ctx_len, D_EMB) but we need to accumulate per (byte, emb_dim)
            # Use np.add.at for scatter-add
            np.add.at(g_emb, xb.reshape(-1), gh_reshaped.reshape(-1, D_EMB))
            g_emb += L2 * emb

            grads = [g_emb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<5i", magic, D_EMB, H, C, ctx_len)
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic(0x454D4245), d, H, C, ctx_len(=16)
 *   float emb[256][d]
 *   float W1[H][ctx_len*d]
 *   float b1[H]
 *   float W2[C][H]
 *   float b2[C]
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], ctx_len = hd[4];
    if (d <= 0 || H <= 0 || C <= 0 || C > 256 || ctx_len != 16) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    size_t expected = 20
        + 256L * d
        + (long)H * ctx_len * d
        + H
        + (long)C * H
        + C;
    expected *= sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * d;
    const float *b1 = W1 + (long)H * ctx_len * d;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)ctx_len) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, ctx_len); return 3; }

        // hidden = b1
        memcpy(hidden, b1, H * sizeof(float));
        // For each context position, add embedding * W1 slice
        for (int j = 0; j < ctx_len; j++) {
            const float *e = emb + xi[j] * d;
            const float *w = W1 + (long)j * d * H;  // W1 is [H][ctx_len*d], so slice per position is [H][d]
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                const float *wh = w + h * ctx_len * d + j * d;  // wait: W1 layout is H x (ctx_len*d)
                // Actually W1 is row-major: H rows, each row has ctx_len*d elements.
                // The slice for position j is d elements at offset j*d within each row.
                const float *wh_row = W1 + h * ctx_len * d + j * d;
                for (int kk = 0; kk < d; kk++) {
                    sum += e[kk] * wh_row[kk];
                }
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // logits = hidden * W2^T + b2
        memcpy(logits, b2, C * sizeof(float));
        for (int c = 0; c < C; c++) {
            const float *wc = W2 + c * H;
            float sum = 0.0f;
            for (int h = 0; h < H; h++) sum += hidden[h] * wc[h];
            logits[c] += sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```