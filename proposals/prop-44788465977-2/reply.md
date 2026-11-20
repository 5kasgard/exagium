```hypothesis
Family: label-rnn (small Elman RNN with 64 hidden units, tanh, trained with Adam).
Why: Character-level language modeling benefits from sequential processing; RNNs capture context order better than bag-of-positions linear/MLP models. The 16-byte context is short enough for efficient unrolled BPTT.
Failure indication: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear.
```
```python
"""label-rnn: Elman RNN for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: embedding(256→64), tanh RNN(64), linear(64→65). Adam lr=3e-3, L2=1e-4, batch=64, epochs≤100.
Weights format: magic 0x524E4E31, d=16, H=64, C=65; float32 arrays: E[256*H], Whh[H*H], bh[H], Why[H*C], by[C]."""
import json, math, struct, sys

import numpy as np

H = 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    V = 256  # byte vocabulary
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Xavier initialization for tanh
    def xavier(fan_in, fan_out, shape):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, shape).astype(np.float32)

    E = xavier(V, H, (V, H))
    Whh = xavier(H, H, (H, H))
    bh = np.zeros(H, np.float32)
    Why = xavier(H, C, (H, C))
    by = np.zeros(C, np.float32)

    params = [E, Whh, bh, Why, by]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Pre-allocate buffers for forward/backward
    max_batch = BATCH
    hs = np.zeros((d, max_batch, H), np.float32)  # hidden states at each time step
    dh = np.zeros((max_batch, H), np.float32)
    dE = np.zeros_like(E)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bsz = len(idx)
            xb = X[idx]          # (bsz, d)
            yb = y[idx]          # (bsz,)

            # Forward pass
            h = np.zeros((bsz, H), np.float32)
            for step in range(d):
                e = E[xb[:, step]]                 # (bsz, H)
                h = np.tanh(e + h @ Whh + bh)      # (bsz, H)
                hs[step, :bsz] = h
            logits = h @ Why + by                  # (bsz, C)

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz                             # dlogits (bsz, C)

            # Gradients for output layer
            dWhy[:] = hs[d-1, :bsz].T @ probs
            dby[:] = probs.sum(axis=0)
            dh[:] = probs @ Why.T                    # (bsz, H)

            # Backprop through time
            dE.fill(0)
            dWhh.fill(0)
            dbh.fill(0)
            for step in reversed(range(d)):
                h = hs[step, :bsz]
                dtanh = dh * (1.0 - h * h)           # (bsz, H)
                # embedding gradient
                for i in range(bsz):
                    dE[xb[i, step]] += dtanh[i]
                # recurrent gradient
                if step > 0:
                    h_prev = hs[step-1, :bsz]
                    dWhh += h_prev.T @ dtanh
                    dh = dtanh @ Whh.T
                else:
                    dWhh += 0  # h_prev is zero
                    dh = dtanh @ Whh.T  # not used after loop
                dbh += dtanh.sum(axis=0)

            # Add L2 regularization gradients
            grads = [dE + L2 * E, dWhh + L2 * Whh, dbh + L2 * bh, dWhy + L2 * Why, dby + L2 * by]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * (gr * gr)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Write weights
    magic = 0x524E4E31  # "RNN1"
    header = struct.pack("<4i", magic, d, H, C)
    blob = header
    for arr in [E, Whh, bh, Why, by]:
        blob += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```
```c
/* label-rnn inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index 0..64). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], H = (int)hd[2], C = (int)hd[3];
    if (d != 16 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)256 * H + (long)H * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *E = p; p += 256 * H;
    const float *Whh = p; p += (long)H * H;
    const float *bh = p; p += H;
    const float *Why = p; p += (long)H * C;
    const float *by = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)calloc(H, sizeof(float));
    float *h_new = (float *)malloc(H * sizeof(float));
    float *logits = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *e = E + xi[step] * H;
            for (int j = 0; j < H; j++) {
                float sum = e[j] + bh[j];
                for (int k2 = 0; k2 < H; k2++) sum += h[k2] * Whh[j * H + k2];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = by[c];
            for (int j = 0; j < H; j++) sum += h[j] * Why[j * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(h_new); free(logits);
    return 0;
}
```