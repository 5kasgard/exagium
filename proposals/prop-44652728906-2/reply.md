```hypothesis
Family: label-embmlp (embedding MLP). Each of the 16 context positions gets a learned 32-dim embedding; these are concatenated and passed through a 64-unit ReLU hidden layer to 65 output classes. This shares statistical strength across positions via the hidden layer while retaining position-specific embeddings, capturing n-gram interactions that linear models miss. Expected to beat label-linear (0.3375) on accuracy with similar or lower CPU cost (embedding lookup + two small dense layers). Failure would be dev accuracy ≤ 0.3375 or Ir/item > 10000.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
E = 32          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
EPOCHS = 10


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
    d = card["input"]["bytes"]  # 16
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward: embedding lookup
            emb_out = np.zeros((B, d, E), np.float32)
            for pos in range(d):
                emb_out[:, pos, :] = emb[pos, xb[:, pos], :]
            flat = emb_out.reshape(B, -1)

            hidden = flat @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Backprop
            grad_W2 = hidden_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_hidden_relu = grad_logits @ W2.T
            grad_hidden = grad_hidden_relu * (hidden > 0)
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(0)
            grad_flat = grad_hidden @ W1.T
            grad_emb_out = grad_flat.reshape(B, d, E)

            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                np.add.at(grad_emb[pos], xb[:, pos], grad_emb_out[:, pos, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic, d, E, H, C, then embeddings, W1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D424D  # "EMBM"
    out = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) hidden[h] = b1[h];

        for (int p = 0; p < d; p++) {
            unsigned char byte = xi[p];
            const float *e = emb + ((long)p * 256 + byte) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) {
                float val = e[e_idx];
                const float *w1_row = W1 + ((long)p * E + e_idx) * H;
                for (int h = 0; h < H; h++) hidden[h] += val * w1_row[h];
            }
        }

        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            const float *w2_row = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2_row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores);
    if (x) free(x);
    free(hd);
    return 0;
}
```