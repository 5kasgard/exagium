```hypothesis
Family: label-rnn (simple recurrent network with shared embeddings).
Reason: Next-character prediction from 16-byte context is a sequence task; an RNN with shared embeddings and recurrent weights captures sequential dependencies better than position-wise linear models, while staying small (E=32, H=64 → ~75 KB weights). The model uses tanh recurrence and a final linear layer, trained with Adam.
Failure signal: Dev accuracy ≤ 0.34 (label-linear level) or inference cost > 150k Ir/item (above label-gbdt) without accuracy gain.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
E = 32          # embedding dimension
H = 64          # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30

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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W_xh = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    W_hh = (rng.uniform(-1, 1, (H, H)) * math.sqrt(6.0 / H)).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_out = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, W_xh, W_hh, b_h, W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward pass
            h = np.zeros((B, H), np.float32)
            h_seq = np.zeros((d, B, H), np.float32)
            for t_step in range(d):
                x_t = emb[xb[:, t_step]]
                h = np.tanh(x_t @ W_xh + h @ W_hh + b_h)
                h_seq[t_step] = h
            logits = h @ W_out + b_out

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs_y = probs[np.arange(B), yb]
            loss = -np.log(probs_y).mean()

            # Backward pass
            dlogits = probs.copy()
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            dW_out = h.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(0) + L2 * b_out
            dh_next = dlogits @ W_out.T

            dW_xh = np.zeros_like(W_xh)
            dW_hh = np.zeros_like(W_hh)
            db_h = np.zeros_like(b_h)
            demb = np.zeros_like(emb)

            for t_step in reversed(range(d)):
                h_t = h_seq[t_step]
                h_prev = np.zeros((B, H), np.float32) if t_step == 0 else h_seq[t_step-1]
                x_t = emb[xb[:, t_step]]
                dtanh = dh_next * (1.0 - h_t * h_t)
                dW_xh += x_t.T @ dtanh
                dW_hh += h_prev.T @ dtanh
                db_h += dtanh.sum(0)
                dx_t = dtanh @ W_xh.T
                np.add.at(demb, xb[:, t_step], dx_t)
                dh_next = dtanh @ W_hh.T

            dW_xh += L2 * W_xh
            dW_hh += L2 * W_hh
            db_h += L2 * b_h
            demb += L2 * emb

            grads = [demb, dW_xh, dW_hh, db_h, dW_out, db_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    magic = 0x524E4E31  # 'RNN1' little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out_bytes = header
    for arr in params:
        out_bytes += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)

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
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = 256L * E + (long)E * H + (long)H * H + H + (long)H * C + C;
    if (nw != 20 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256L * E;
    const float *W_xh = p; p += (long)E * H;
    const float *W_hh = p; p += (long)H * H;
    const float *b_h = p; p += H;
    const float *W_out = p; p += (long)H * C;
    const float *b_out = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int t = 0; t < d; t++) {
            const float *e = emb + xi[t] * E;
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int kk = 0; kk < E; kk++) sum += e[kk] * W_xh[kk * H + j];
                for (int kk = 0; kk < H; kk++) sum += h[kk] * W_hh[kk * H + j];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
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
    free(h); free(h_new); free(logits);
    return 0;
}
```