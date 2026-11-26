```hypothesis
Family: label-rnn (small Elman RNN with byte embeddings).
Why: The task is next-character prediction from 16-byte context. A tiny RNN can capture sequential dependencies (e.g., "th"→"e") that a linear model misses, while staying extremely lightweight (≈5k parameters). Embedding each byte (256→8) then processing with a 32-unit tanh RNN and a final linear layer should exceed label-linear's 33.75% accuracy.
Failure: Dev accuracy ≤ 33.75% or inference cost > 2× label-linear (≈15k Ir/item).
```

```python
"""label-rnn: tiny Elman RNN for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, numpy only.
Architecture: 16-step unrolled RNN. Byte -> embed(8) -> RNN(32, tanh) -> Linear(65).
Exported weights: magic 0x524E4E31, d=16, E=8, H=32, C=65; then float32 arrays in order:
  emb[256][E], Wxh[E][H], Whh[H][H], bh[H], Why[H][C], by[C].
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split)
E = 8        # embedding dimension
H = 32       # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30
CLIP = 5.0   # gradient clipping

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

    rng = np.random.default_rng(SEED)
    # He-ish initialization
    emb = (rng.normal(0, 1/math.sqrt(E), (256, E))).astype(np.float32)
    Wxh = (rng.normal(0, 1/math.sqrt(E), (E, H))).astype(np.float32)
    Whh = (rng.normal(0, 1/math.sqrt(H), (H, H))).astype(np.float32)
    bh  = np.zeros(H, np.float32)
    Why = (rng.normal(0, 1/math.sqrt(H), (H, C))).astype(np.float32)
    by  = np.zeros(C, np.float32)

    params = [emb, Wxh, Whh, bh, Why, by]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward_batch(xb):
        """xb: (B, d) uint8 -> returns logits (B, C), and caches for bptt"""
        B = xb.shape[0]
        h = np.zeros((B, H), np.float32)
        # caches: list of (x_emb, h_pre, h_post) per step
        caches = []
        for step in range(d):
            x_emb = emb[xb[:, step]]          # (B, E)
            h_pre = x_emb @ Wxh + h @ Whh + bh
            h_post = np.tanh(h_pre)
            caches.append((x_emb, h_pre, h_post, h.copy()))
            h = h_post
        logits = h @ Why + by
        return logits, caches

    def loss_grad(logits, yb):
        """Cross-entropy loss and dlogits"""
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(B), yb] + 1e-12).mean()
        dlogits = probs
        dlogits[np.arange(B), yb] -= 1.0
        dlogits /= B
        return loss, dlogits

    def backward_batch(dlogits, caches, yb):
        """BPTT through 16 steps. Returns grads for each param."""
        B = dlogits.shape[0]
        dWhy = caches[-1][3].T @ dlogits      # last h (B,H)^T @ dlogits (B,C) -> (H,C)
        dby  = dlogits.sum(axis=0)
        dh = dlogits @ Why.T                  # (B,H)

        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dbh  = np.zeros_like(bh)
        demb = np.zeros_like(emb)

        # iterate steps backwards
        for step in reversed(range(d)):
            x_emb, h_pre, h_post, h_prev = caches[step]
            dtanh = dh * (1 - h_post * h_post)   # (B,H)
            dbh += dtanh.sum(axis=0)
            dWxh += x_emb.T @ dtanh
            dWhh += h_prev.T @ dtanh
            demb_flat = dtanh @ Wxh.T              # (B,E)
            # accumulate into demb via scatter-add
            idx = Xb[:, step]                      # (B,)
            np.add.at(demb, idx, demb_flat)
            dh = dtanh @ Whh.T

        # L2 regularization gradients
        dWxh += L2 * Wxh
        dWhh += L2 * Whh
        dWhy += L2 * Why
        # no L2 on biases/embeddings (common practice)
        return [demb, dWxh, dWhh, dbh, dWhy, dby]

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            logits, caches = forward_batch(Xb)
            loss, dlogits = loss_grad(logits, yb)
            grads = backward_batch(dlogits, caches, yb)

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            epoch_loss += loss
            batches += 1
        # Optional: early stopping could be added, but fixed epochs for determinism.

    # Pack weights
    # magic, d, E, H, C
    header = struct.pack("<5i", 0x524E4E31, d, E, H, C)
    body = (emb.astype("<f4").tobytes() +
            Wxh.astype("<f4").tobytes() +
            Whh.astype("<f4").tobytes() +
            bh.astype("<f4").tobytes() +
            Why.astype("<f4").tobytes() +
            by.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-rnn inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x524E4E31, d, E, H, C; then float32 arrays:
 *   emb[256][E], Wxh[E][H], Whh[H][H], bh[H], Why[H][C], by[C].
 * C11, libc+libm only. No dynamic allocation per item. */
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
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    // Expected weight size: header(20) + arrays
    long expected = 20
        + 4L * (256 * E)        // emb
        + 4L * (E * H)          // Wxh
        + 4L * (H * H)          // Whh
        + 4L * H                // bh
        + 4L * (H * C)          // Why
        + 4L * C;               // by
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *Wxh = p; p += E * H;
    const float *Whh = p; p += H * H;
    const float *bh  = p; p += H;
    const float *Why = p; p += H * C;
    const float *by  = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    float *x_emb = malloc(E * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            // x_emb = emb[xi[step]]
            const float *emb_row = emb + xi[step] * E;
            memcpy(x_emb, emb_row, E * sizeof(float));

            // h_new = tanh(x_emb @ Wxh + h @ Whh + bh)
            for (int j = 0; j < H; j++) {
                float sum = bh[j];
                for (int e = 0; e < E; e++) sum += x_emb[e] * Wxh[e * H + j];
                for (int hh = 0; hh < H; hh++) sum += h[hh] * Whh[hh * H + j];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }

        // logits = h @ Why + by
        for (int c = 0; c < C; c++) {
            float sum = by[c];
            for (int j = 0; j < H; j++) sum += h[j] * Why[j * C + c];
            logits[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(h_new); free(logits); free(x_emb);
    if (x) free(x);
    return 0;
}
```