```hypothesis
Family: label-embmlp. Uses a shared per-position embedding (16-dim) for each of the 16 context bytes, concatenates them (256-dim), then a ReLU hidden layer (64 units) and a linear output (65 classes). This captures pairwise position interactions via the hidden layer and learns character similarities via embeddings, which the linear model cannot. Expected to beat label-linear (0.3375) on dev. Failure if dev accuracy ≤ 0.3375 or inference cost > 2× label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
E = 16          # embedding size per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embeddings = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    W1 = rng.normal(0, math.sqrt(2.0 / (d*E + H)), (d*E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / (H + C)), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            # Forward: gather embeddings
            emb_concat = np.zeros((B, d*E), np.float32)
            for pos in range(d):
                emb_concat[:, pos*E:(pos+1)*E] = embeddings[pos][x_batch[:, pos]]

            hidden_pre = emb_concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)

            logits = hidden @ W2 + b2

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = probs / B

            # Backprop
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0

            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)

            grad_W1 = emb_concat.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            grad_embeddings = [np.zeros_like(e) for e in embeddings]
            for pos in range(d):
                g = grad_hidden[:, pos*E:(pos+1)*E]
                np.add.at(grad_embeddings[pos], x_batch[:, pos], g)

            # L2 regularization on weights (not biases)
            for i, p in enumerate(params):
                if i < d or i == d or i == d+2:  # embeddings, W1, W2
                    if i < d:
                        grad_embeddings[i] += L2 * p
                    elif i == d:
                        grad_W1 += L2 * p
                    elif i == d+2:
                        grad_W2 += L2 * p

            grads = grad_embeddings + [grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x454D424D  # "EMBM"
    header = struct.pack("<5i", magic, d, E, H, C)
    emb_data = b""
    for pos in range(d):
        emb_data += embeddings[pos].astype("<f4").tobytes()
    w1_data = W1.astype("<f4").tobytes()
    b1_data = b1.astype("<f4").tobytes()
    w2_data = W2.astype("<f4").tobytes()
    b2_data = b2.astype("<f4").tobytes()

    out = header + emb_data + w1_data + b1_data + w2_data + b2_data
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + (long)d * 256 * E * 4 + (long)d * E * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int p = 0; p < d; p++) {
            const float *e = emb + (long)p * 256 * E + xi[p] * E;
            memcpy(concat + p * E, e, E * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d * E; j++) {
                sum += concat[j] * W1[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(concat);
    free(hidden);
    free(logits);
    return 0;
}
```