```hypothesis
Family: label-rnn (small recurrent network with 32-dim embeddings, 64-unit tanh hidden state).
Why: Character prediction benefits from sequential processing; RNN captures local dependencies better than position-independent linear model.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× linear.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on the last 20% of train.in)
E = 32          # embedding dimension
H = 64          # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-5
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
    assert d == 16
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float64 for training stability)
    embed = rng.uniform(-0.1, 0.1, (256, E)).astype(np.float64)
    W_xh = rng.uniform(-math.sqrt(6.0 / E), math.sqrt(6.0 / E), (E, H)).astype(np.float64)
    W_hh = rng.uniform(-math.sqrt(6.0 / H), math.sqrt(6.0 / H), (H, H)).astype(np.float64)
    b_h = np.zeros(H, np.float64)
    W_hy = rng.uniform(-math.sqrt(6.0 / H), math.sqrt(6.0 / H), (H, C)).astype(np.float64)
    b_y = np.zeros(C, np.float64)

    params = [embed, W_xh, W_hh, b_h, W_hy, b_y]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = x_batch.shape[0]

            # Forward pass
            h = np.zeros((B, H), np.float64)
            h_states = np.zeros((d + 1, B, H), np.float64)
            h_states[0] = h
            for step in range(d):
                emb = embed[x_batch[:, step]]
                pre = emb @ W_xh + h @ W_hh + b_h
                h = np.tanh(pre)
                h_states[step + 1] = h

            logits = h @ W_hy + b_y

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), y_batch] -= 1.0
            dlogits /= B

            # Output layer gradients
            dW_hy = h.T @ dlogits + L2 * W_hy
            db_y = dlogits.sum(axis=0)

            # Backprop through time
            dh = dlogits @ W_hy.T
            dW_xh = np.zeros_like(W_xh)
            dW_hh = np.zeros_like(W_hh)
            db_h = np.zeros_like(b_h)
            dembed = np.zeros_like(embed)

            for step in reversed(range(d)):
                h_prev = h_states[step]
                h_cur = h_states[step + 1]
                dtanh = dh * (1.0 - h_cur * h_cur)
                emb = embed[x_batch[:, step]]
                dW_xh += emb.T @ dtanh
                dW_hh += h_prev.T @ dtanh
                db_h += dtanh.sum(axis=0)
                dembed_batch = dtanh @ W_xh.T
                np.add.at(dembed, x_batch[:, step], dembed_batch)
                dh = dtanh @ W_hh.T

            dW_xh += L2 * W_xh
            dW_hh += L2 * W_hh
            dembed += L2 * embed

            grads = [dembed, dW_xh, dW_hh, db_h, dW_hy, db_y]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Convert to float32 and write weights.bin
    embed_f32 = embed.astype(np.float32)
    W_xh_f32 = W_xh.astype(np.float32)
    W_hh_f32 = W_hh.astype(np.float32)
    b_h_f32 = b_h.astype(np.float32)
    W_hy_f32 = W_hy.astype(np.float32)
    b_y_f32 = b_y.astype(np.float32)

    magic = 0x524E4E31  # "RNN1"
    header = struct.pack("<5i", magic, E, H, C, d)
    out = header
    out += embed_f32.tobytes()
    out += W_xh_f32.tobytes()
    out += W_hh_f32.tobytes()
    out += b_h_f32.tobytes()
    out += W_hy_f32.tobytes()
    out += b_y_f32.tobytes()

    open(sys.argv[4], "wb").write(out)

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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int E = hd[1], H = hd[2], C = hd[3], d = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20
        + (long)256 * E * 4
        + (long)E * H * 4
        + (long)H * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *embed = (float *)(hd + 5);
    float *W_xh = embed + 256 * E;
    float *W_hh = W_xh + (long)E * H;
    float *b_h = W_hh + (long)H * H;
    float *W_hy = b_h + H;
    float *b_y = W_hy + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *emb = malloc(E * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *e = embed + xi[step] * E;
            memcpy(emb, e, E * sizeof(float));
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k = 0; k < E; k++) sum += emb[k] * W_xh[k * H + j];
                for (int k = 0; k < H; k++) sum += h[k] * W_hh[k * H + j];
                h_new[j] = tanhf(sum);
            }
            float *tmp = h; h = h_new; h_new = tmp;
        }
        for (int c = 0; c < C; c++) {
            float sum = b_y[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_hy[j * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```