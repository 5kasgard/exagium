```hypothesis
Family: label-emb-mlp (position-specific embeddings + one ReLU layer).
Why: The task is next-character prediction from 16 bytes. Linear model uses one-hot per position (4096 features). Position-specific embeddings reduce dimensionality and capture character similarities, while a hidden ReLU layer adds non-linearity. This should beat linear accuracy (0.3375) with moderate inference cost (table lookup + small matmul). If dev accuracy ≤ 0.3375 or inference cost > 2× linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
E = 32          # embedding size per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
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
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]   # 16
    C = card["classes"]          # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Position-specific embeddings: list of d arrays (256, E)
    embeddings = [rng.normal(0, 1.0 / math.sqrt(E), size=(256, E)).astype(np.float32) for _ in range(d)]
    # Position-specific input->hidden weights: list of d arrays (E, H)
    W1_list = [rng.normal(0, math.sqrt(2.0 / (E + H)), size=(E, H)).astype(np.float32) for _ in range(d)]
    b1 = np.zeros(H, dtype=np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / (H + C)), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = embeddings + W1_list + [b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            h_pre = np.tile(b1, (B, 1))           # (B, H)
            embs = []
            for j in range(d):
                emb_j = embeddings[j][Xb[:, j]]   # (B, E)
                embs.append(emb_j)
                h_pre += emb_j @ W1_list[j]       # (B, H)
            h = np.maximum(h_pre, 0)              # ReLU
            logits = h @ W2 + b2                  # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # Backprop
            dh = dlogits @ W2.T                   # (B, H)
            dh_pre = dh * (h_pre > 0)             # (B, H)

            db1 = dh_pre.sum(axis=0)
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)

            dembeddings = [np.zeros_like(emb) for emb in embeddings]
            dW1_list = [np.zeros_like(W1) for W1 in W1_list]

            for j in range(d):
                emb_j = embs[j]
                dW1_list[j] = emb_j.T @ dh_pre
                demb_j = dh_pre @ W1_list[j].T    # (B, E)
                indices = Xb[:, j]
                for e in range(E):
                    np.add.at(dembeddings[j][:, e], indices, demb_j[:, e])

            # L2 regularization
            for j in range(d):
                dW1_list[j] += L2 * W1_list[j]
                dembeddings[j] += L2 * embeddings[j]
            dW2 += L2 * W2

            grads = dembeddings + dW1_list + [db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build inference table: Table1[d][256][H] = embeddings[j][b] @ W1_list[j]
    Table1 = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        Table1[j] = embeddings[j] @ W1_list[j]    # (256, H)

    # Pack weights: magic, d, H, C; Table1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, H, C)
    out += Table1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *Table1 = (const float *)(hd + 4);
    const float *b1 = Table1 + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(hidden, b1, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *t = Table1 + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        memcpy(scores, b2, sizeof(float) * C);
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```