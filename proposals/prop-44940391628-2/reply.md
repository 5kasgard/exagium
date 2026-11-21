```hypothesis
We propose a "label-rnn" family: a small Elman RNN with shared embeddings across the 16‑byte context.  
The recurrent structure captures sequential dependencies better than the position‑wise linear model, while parameter sharing keeps the model tiny (≈16 k parameters) and inference fast.  
If dev accuracy does not exceed label‑linear (0.3375) or the instruction count per item is significantly higher, the family is not worthwhile.
```

```python
"""label-rnn: simple Elman RNN with shared embeddings for 16‑byte context.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, not tuned on the tuning split)
EMBED_DIM = 32
HIDDEN_SIZE = 32
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CLIP_GRAD = 1.0

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
    n, d = X.shape
    C = card["classes"]
    V = 256                     # input byte vocabulary
    rng = np.random.default_rng(SEED)

    # Xavier initialization
    def xavier(in_dim, out_dim):
        return (rng.uniform(-1, 1, (out_dim, in_dim)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    def zero(shape):
        return np.zeros(shape, np.float32)

    E = xavier(V, EMBED_DIM)                # (V, E)
    W_xh = xavier(EMBED_DIM, HIDDEN_SIZE)   # (H, E)
    W_hh = xavier(HIDDEN_SIZE, HIDDEN_SIZE) # (H, H)
    b_h = zero(HIDDEN_SIZE)                 # (H,)
    W_out = xavier(HIDDEN_SIZE, C)          # (C, H)
    b_out = zero(C)                         # (C,)

    params = [E, W_xh, W_hh, b_h, W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]          # (B, 16)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward pass – store intermediates for BPTT
            h = np.zeros((B, HIDDEN_SIZE), np.float32)
            hs = [h.copy()]           # hs[0] = 0, hs[t] after step t
            embs = []                 # embeddings used at each step

            for step in range(d):
                x_t = x_batch[:, step]                 # (B,)
                e_t = E[x_t]                           # (B, E)
                embs.append(e_t)
                h = np.tanh(e_t @ W_xh.T + h @ W_hh.T + b_h)
                hs.append(h.copy())

            # Output
            logits = h @ W_out.T + b_out                # (B, C)
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Cross-entropy gradient
            dlogits = probs.copy()
            dlogits[np.arange(B), y_batch] -= 1.0
            dlogits /= B

            # Backward pass
            grads = [np.zeros_like(p) for p in params]
            # Output layer
            grads[4] = dlogits.T @ h          # dW_out (C, H)
            grads[5] = dlogits.sum(axis=0)    # db_out (C,)
            dh_next = dlogits @ W_out         # (B, H)

            # BPTT
            for step in reversed(range(d)):
                h_t = hs[step + 1]
                h_prev = hs[step]
                e_t = embs[step]
                x_t = x_batch[:, step]

                # Gradient through tanh
                dtanh = (1 - h_t * h_t) * dh_next          # (B, H)

                # Gradients for W_xh, W_hh, b_h
                grads[1] += dtanh.T @ e_t                   # dW_xh
                grads[2] += dtanh.T @ h_prev                # dW_hh
                grads[3] += dtanh.sum(axis=0)               # db_h

                # Propagate to previous hidden and embedding
                de_t = dtanh @ W_xh                         # (B, E)
                dh_prev = dtanh @ W_hh                      # (B, H)

                # Scatter embedding gradients
                np.add.at(grads[0], x_t, de_t)

                dh_next = dh_prev

            # L2 regularization
            for i, p in enumerate(params):
                grads[i] += L2 * p

            # Gradient clipping
            total_norm = 0.0
            for g in grads:
                total_norm += np.sum(g * g)
            total_norm = math.sqrt(total_norm)
            if total_norm > CLIP_GRAD:
                scale = CLIP_GRAD / total_norm
                for g in grads:
                    g *= scale

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic, d, E, H, C, V, then float32 arrays in order
    out = struct.pack("<6i", 0x524E4E31, d, EMBED_DIM, HIDDEN_SIZE, C, V)
    out += E.astype("<f4").tobytes()
    out += W_xh.astype("<f4").tobytes()
    out += W_hh.astype("<f4").tobytes()
    out += b_h.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-rnn: simple Elman RNN inference.
 * ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4], V = hd[5];
    if (d != 16 || V != 256 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 24 + 4L * ( (long)V * E + (long)H * E + (long)H * H + H + (long)C * H + C );
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 6);
    const float *emb = p; p += V * E;
    const float *W_xh = p; p += H * E;
    const float *W_hh = p; p += H * H;
    const float *b_h = p; p += H;
    const float *W_out = p; p += C * H;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *e_t = malloc(E * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    float *tmp = malloc(H * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *emb_row = emb + xi[step] * E;
            memcpy(e_t, emb_row, E * sizeof(float));
            // tmp = e_t @ W_xh.T + h @ W_hh.T + b_h
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k = 0; k < E; k++) sum += e_t[k] * W_xh[j * E + k];
                for (int k = 0; k < H; k++) sum += h[k] * W_hh[j * H + k];
                tmp[j] = tanhf(sum);
            }
            memcpy(h, tmp, H * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_out[c * H + j];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```