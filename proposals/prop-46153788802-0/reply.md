```hypothesis
Family: label-emb (embedding-based MLP). Each of the 14 features gets a learned 8-dim embedding (256 values per feature). Concatenated embeddings (112 dim) feed a single ReLU hidden layer of 32 units, then a 2-class output. This captures non-linear feature interactions at lower parameter count than one-hot linear, and should beat label-linear's 0.8524 accuracy while staying fast (≈1k Ir/item). Failure: dev accuracy ≤0.8524 or Ir/item >3000.
```

```python
"""label-emb: embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Embedding size 8, hidden 32, Adam lr=3e-3, L2=1e-4, batch 64, epochs min(100, ceil(600000/n)).
Weights layout (little-endian):
  uint32 magic=0x4C424D45, d, k, H, C
  float32 emb[d][256][k]
  float32 W1[d*k][H], b1[H]
  float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

K, H, SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 8, 32, 1, 3e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # embeddings: d x 256 x K
    emb = (rng.normal(0, 1/math.sqrt(K), (d, 256, K))).astype(np.float32)
    # MLP
    W1 = (rng.normal(0, math.sqrt(2.0/(d*K)), (d*K, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx_all = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, n, BATCH):
            idx = idx_all[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            e = np.concatenate([emb[j, xb[:, j]] for j in range(d)], axis=1)  # (bs, d*K)
            h = np.maximum(e @ W1 + b1, 0)  # (bs, H)
            logits = h @ W2 + b2  # (bs, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs  # (bs, C)

            # backward
            grad_W2 = h.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            dh = probs @ W2.T * (h > 0)
            grad_W1 = e.T @ dh + L2 * W1
            grad_b1 = dh.sum(axis=0)
            de = dh @ W1.T  # (bs, d*K)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                off = j * K
                np.add.at(grad_emb[j], xb[:, j], de[:, off:off+K])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t = 1  # simple Adam step counter per batch (approximate)
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    out = struct.pack("<5i", 0x4C424D45, d, K, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * K + (long)d * K * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * K;
    const float *b1 = W1 + (long)d * K * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * K);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            memcpy(concat + j * K, e, K * sizeof(float));
        }

        // hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * K; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```