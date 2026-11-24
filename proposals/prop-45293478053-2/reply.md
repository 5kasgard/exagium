```hypothesis
Family: label-rnn (small Elman RNN with shared byte embeddings).
Why: Character-level language modeling benefits from sequential processing; an RNN shares parameters across the 16 context positions and captures n-gram statistics via its hidden state. With embedding dim 16 and hidden size 32 it has ~10k parameters (far fewer than label-linear's 4096×65) and ~27k FLOPs/item, so it should be both more accurate and faster.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item ≥ 7359.
```
```python
"""label-rnn: Elman RNN with shared byte embeddings for fixed-length byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only.
Weights format (little-endian):
  int32 magic=0x524E4E31, d=16, D, H, C
  float32 embedding[256*D], W_x[H*D], W_h[H*H], b_h[H], W_out[C*H], b_out[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES = 3e-3, 1e-4, 64, 600000
D, H = 16, 32          # embedding dim, hidden size (tuned on last 20% of train)
MAX_EPOCHS = 100

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
    C = card["classes"]
    n, d = X.shape
    assert d == 16

    rng = np.random.default_rng(SEED)
    # init parameters
    scale = lambda n: math.sqrt(6.0 / n)
    emb = (rng.uniform(-1, 1, (256, D)) * scale(256 + D)).astype(np.float32)
    W_x = (rng.uniform(-1, 1, (H, D)) * scale(D + H)).astype(np.float32)
    W_h = (rng.uniform(-1, 1, (H, H)) * scale(H + H)).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_out = (rng.uniform(-1, 1, (C, H)) * scale(H + C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, W_x, W_h, b_h, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, 16) uint8
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # forward
            emb_x = emb[xb]      # (B, 16, D)
            h = np.zeros((B, H), np.float32)
            h_list = [h.copy()]
            for step in range(16):
                x_t = emb_x[:, step, :]
                h = np.tanh(x_t @ W_x.T + h @ W_h.T + b_h)
                h_list.append(h.copy())
            logits = h @ W_out.T + b_out

            # loss & dlogits
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # gradients
            dW_out = dlogits.T @ h_list[-1]
            db_out = dlogits.sum(axis=0)
            dh_next = dlogits @ W_out

            dW_x = np.zeros_like(W_x)
            dW_h = np.zeros_like(W_h)
            db_h = np.zeros_like(b_h)
            demb = np.zeros_like(emb)

            for step in reversed(range(16)):
                h_t = h_list[step + 1]
                h_prev = h_list[step]
                x_t = emb_x[:, step, :]
                dtanh = (1.0 - h_t * h_t) * dh_next

                dW_x += dtanh.T @ x_t
                dW_h += dtanh.T @ h_prev
                db_h += dtanh.sum(axis=0)

                dx_t = dtanh @ W_x
                np.add.at(demb, xb[:, step], dx_t)

                dh_next = dtanh @ W_h

            grads = [demb, dW_x, dW_h, db_h, dW_out, db_out]

            # L2
            for p, g in zip(params, grads):
                g += L2 * p

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic, d, D_, H_, C_ = 0x524E4E31, d, D, H, C
    expected = 16 + 4 * (256 * D + H * D + H * H + H + C * H + C)
    out = struct.pack("<5i", magic, d, D, H, C)
    out += emb.astype("<f4").tobytes()
    out += W_x.astype("<f4").tobytes()
    out += W_h.astype("<f4").tobytes()
    out += b_h.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    assert len(out) == expected, f"size mismatch {len(out)} vs {expected}"
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rnn inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * (256L * D + (long)H * D + (long)H * H + H + (long)C * H + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *embedding = p; p += 256 * D;
    const float *W_x = p; p += H * D;
    const float *W_h = p; p += H * H;
    const float *b_h = p; p += H;
    const float *W_out = p; p += C * H;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *tmp = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < 16; step++) {
            const float *emb = embedding + xi[step] * D;
            /* tmp = emb * W_x^T + h * W_h^T + b_h */
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k = 0; k < D; k++) sum += emb[k] * W_x[j * D + k];
                for (int k = 0; k < H; k++) sum += h[k] * W_h[j * H + k];
                tmp[j] = sum;
            }
            for (int j = 0; j < H; j++) h_new[j] = tanhf(tmp[j]);
            memcpy(h, h_new, H * sizeof(float));
        }
        /* logits = h * W_out^T + b_out */
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_out[c * H + j];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(h_new); free(tmp); free(logits);
    return 0;
}
```