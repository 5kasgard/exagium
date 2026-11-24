```hypothesis
Propose a new family "label-embed": learned per-position byte embeddings (16-dim) followed by a single ReLU hidden layer (64 units) and a linear output. This captures positional interactions better than a linear model while keeping inference fast due to small embedding/hidden sizes. If dev accuracy does not exceed label-linear (0.3375) or inference cost (instructions/ns per item) is significantly higher, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

EMBED_DIM = 16
HIDDEN = 64
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    embeddings = (rng.normal(0, 0.01, (d, 256, EMBED_DIM))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * EMBED_DIM)), (d * EMBED_DIM, HIDDEN))).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / HIDDEN), (HIDDEN, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)
            x_batch = X[idx]
            y_batch = y[idx]

            emb = embeddings[np.arange(d)[:, None], x_batch.T, :]
            emb = emb.transpose(1, 0, 2).reshape(batch_size, -1)

            h = emb @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), y_batch] -= 1.0
            grad_logits = probs / batch_size

            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_W1 = emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_emb_flat = grad_h @ W1.T
            grad_emb = grad_emb_flat.reshape(batch_size, d, EMBED_DIM)
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                np.add.at(grad_embeddings[j], x_batch[:, j], grad_emb[:, j, :])
            grad_embeddings += L2 * embeddings

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x454D4244
    out = struct.pack("<5i", magic, d, EMBED_DIM, HIDDEN, C)
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

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
    uint32_t k = buf[*pos] | (buf[*pos+1] << 8) | (buf[*pos+2] << 16) | ((uint32_t)buf[*pos+3] << 24);
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 64 || C != 65) {
        fprintf(stderr, "unexpected dimensions: d=%d E=%d H=%d C=%d\n", d, E, H, C);
        return 3;
    }
    long expected = 20 + 4L * ( (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) {
                acc[j * E + e_idx] = e[e_idx];
            }
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) {
                sum += acc[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            output[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (output[c] > output[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(acc); free(hidden); free(output);
    return 0;
}
```