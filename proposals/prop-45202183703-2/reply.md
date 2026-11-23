```hypothesis
Family: label-rnn (tiny recurrent network over the 16-byte context).
Why: Character-level language modeling benefits from sequential processing; a small RNN with shared+positional embeddings captures local dependencies better than a linear model while staying extremely compact (≈50 KiB weights) and fast (≈2 μs/item).
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference > 1500 ns/item.
```

```python
"""label-rnn: tiny RNN for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: shared byte embedding (256×H) + positional embedding (16×H) → RNN (H hidden, tanh) → linear (H→65).
H=32, Adam lr=1e-3, batch=64, epochs=30. Weights exported as float32 little-endian."""
import json, math, struct, sys

import numpy as np

H = 32
SEED, LR, BATCH, EPOCHS = 1, 1e-3, 64, 30
MAGIC = 0x524E4E31  # "RNN1"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # Parameters
    W_emb = (rng.normal(0, 0.02, (256, H))).astype(np.float32)
    W_pos = (rng.normal(0, 0.02, (d, H))).astype(np.float32)
    W_hh = (rng.normal(0, 0.02, (H, H)) * math.sqrt(1.0 / H)).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_out = (rng.normal(0, 0.02, (H, C)) * math.sqrt(1.0 / H)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [W_emb, W_pos, W_hh, b_h, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    order = np.arange(n)
    for ep in range(EPOCHS):
        rng.shuffle(order)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # Forward
            h = np.zeros((bsz, H), np.float32)
            for t in range(d):
                emb = W_emb[xb[:, t]] + W_pos[t]
                h = np.tanh(emb + h @ W_hh + b_h)
            logits = h @ W_out + b_out
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz
            # Backward
            gW_out = h.T @ probs
            gb_out = probs.sum(axis=0)
            gh = probs @ W_out.T
            gW_hh = np.zeros((H, H), np.float32)
            gb_h = np.zeros(H, np.float32)
            gW_emb = np.zeros((256, H), np.float32)
            gW_pos = np.zeros((d, H), np.float32)
            for t in reversed(range(d)):
                dt = gh * (1 - h * h)
                gb_h += dt.sum(axis=0)
                gW_hh += h.T @ dt if t == d - 1 else h_prev.T @ dt
                gW_emb[xb[:, t]] += dt
                gW_pos[t] += dt.sum(axis=0)
                gh = dt @ W_hh.T
                h_prev = h if t == 0 else np.tanh(W_emb[xb[:, t-1]] + W_pos[t-1] + (h if t==1 else h_prev) @ W_hh + b_h)
                h = h_prev
            grads = [gW_emb, gW_pos, gW_hh, gb_h, gW_out, gb_out]
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * m / (np.sqrt(v) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, H, C)
    for p in params:
        out += p.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rnn inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic(0x524E4E31), d, H, C; then float32 arrays in order:
 *   W_emb[256*H], W_pos[d*H], W_hh[H*H], b_h[H], W_out[H*C], b_out[C] */
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
    if (nw < 16 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H != 32) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (256L*H + d*H + (long)H*H + H + (long)H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_emb = p; p += 256 * H;
    const float *W_pos = p; p += d * H;
    const float *W_hh  = p; p += H * H;
    const float *b_h   = p; p += H;
    const float *W_out = p; p += H * C;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    float *tmp = malloc(sizeof(float) * H);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int t = 0; t < d; t++) {
            const float *emb = W_emb + xi[t] * H;
            const float *pos_emb = W_pos + t * H;
            for (int j = 0; j < H; j++) tmp[j] = emb[j] + pos_emb[j] + h[j];
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k2 = 0; k2 < H; k2++) sum += tmp[k2] * W_hh[k2 * H + j];
                h[j] = tanhf(sum);
            }
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_out[j * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```