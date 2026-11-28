```hypothesis
Family: label-embed (per-position byte embeddings + 1-hidden-layer ReLU MLP).
Why: The task is next-character prediction from 16 bytes of context. Label-linear uses a huge one-hot encoding (16×256 features) with a linear classifier, which cannot model interactions between positions. Learned embeddings (16 tables of 256×D) compress the input into a dense representation, and a small ReLU layer captures position interactions. With D=16, H=64 the model has ~86k parameters (vs 266k for label-linear) and should achieve higher accuracy at similar or lower inference cost.
Failure signal: Dev accuracy ≤ 0.35 (does not beat label-linear's 0.3375 by a meaningful margin).
```
```python
"""label-embed: per-position byte embeddings + 1-hidden-layer ReLU MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x45424445 ('EMBE'), d=16, D, H, C
  float32 emb[d][256][D]         (row-major: position, byte, dim)
  float32 W1[d*D][H], b1[H]
  float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2 = 3e-3, 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
D = 16          # embedding dimension per position
H = 64          # hidden layer size


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
    assert d == 16

    rng = np.random.default_rng(SEED)

    # Embeddings: (d, 256, D)
    emb = rng.normal(0, 1.0 / math.sqrt(D), size=(d, 256, D)).astype(np.float32)

    # MLP weights
    in_dim = d * D
    W1 = rng.uniform(-1, 1, (in_dim, H)).astype(np.float32) * math.sqrt(6.0 / in_dim)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward: embed -> concat -> hidden -> output
            # emb_lookup: (bs, d, D)
            emb_lookup = emb[xb]                    # (bs, d, D)
            flat = emb_lookup.reshape(bs, in_dim)   # (bs, d*D)
            hidden_pre = flat @ W1 + b1             # (bs, H)
            hidden = np.maximum(hidden_pre, 0)      # ReLU
            logits = hidden @ W2 + b2               # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = probs / bs                   # (bs, C)

            # Backprop
            grad_W2 = hidden.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_hidden = grad_out @ W2.T           # (bs, H)
            grad_hidden[hidden_pre <= 0] = 0.0      # ReLU grad
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_flat = grad_hidden @ W1.T          # (bs, d*D)
            grad_emb_lookup = grad_flat.reshape(bs, d, D)  # (bs, d, D)

            # Accumulate embedding gradients
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, (xb[:, :, None], np.arange(D)[None, None, :]), grad_emb_lookup)
            # The above is wrong; we need to index by position and byte value.
            # Let's do it correctly with a loop over positions (d=16 is small).
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                bytes_j = xb[:, j]          # (bs,)
                # For each position j, add grad_emb_lookup[:, j, :] to grad_emb[j, bytes_j, :]
                np.add.at(grad_emb[j], bytes_j, grad_emb_lookup[:, j, :])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x45424445  # 'EMBE'
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x45424445, d, D, H, C
 *   float32 emb[d][256][D]
 *   float32 W1[d*D][H], b1[H]
 *   float32 W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x45424445) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D <= 0 || H <= 0 || C <= 0 || C > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * D;
    size_t w1_sz = (size_t)d * D * H;
    size_t w2_sz = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + w1_sz + H + w2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + w1_sz;
    const float *W2 = b1 + H;
    const float *b2 = W2 + w2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int m = 0; m < d * D; m++) sum += concat[m] * W1[(size_t)m * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```