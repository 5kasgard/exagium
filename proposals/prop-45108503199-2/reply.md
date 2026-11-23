```hypothesis
Family: label-sharemb (shared character embeddings + 1-hidden-layer MLP).
Why: Character identity is position-invariant; sharing a 32-dim embedding table across all 16 positions reduces parameters (vs per-position one-hot) and captures character similarities, while a 128-unit ReLU layer models position interactions. This should generalize better than label-linear on 24k examples.
Failure signal: dev accuracy ≤ 0.34 (does not beat label-linear's 0.3375).
```
```python
"""label-sharemb: Shared character embeddings + 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: E[256][D] shared embeddings -> concat 16 positions (16*D) -> ReLU(H) -> Linear(C).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(80, ceil(500000/n)). Deterministic seed=1.
Export: magic 0x53484D42, d=16, D, H, C; float32 E[256][D], W1[16*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

D, H = 32, 128
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 500000, 80


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
    d = X.shape[1]          # 16
    rng = np.random.default_rng(SEED)

    # Xavier init
    E = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            emb = E[xb]                     # (bs, d, D)
            flat = emb.reshape(bs, d * D)   # (bs, d*D)
            h_pre = flat @ W1 + b1          # (bs, H)
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2            # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs                     # (bs, C)

            # Backprop
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0) + L2 * b2
            gh = probs @ W2.T
            gh[h_pre <= 0] = 0.0
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0) + L2 * b1
            gflat = gh @ W1.T               # (bs, d*D)
            gemb = gflat.reshape(bs, d, D)  # (bs, d, D)

            # Sparse update for E: accumulate per (batch, pos, char)
            gE = np.zeros_like(E)
            np.add.at(gE, xb, gemb)         # xb is (bs, d), gemb is (bs, d, D) -> broadcasts correctly
            gE += L2 * E

            grads = [gE, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, D, H, C; then E, W1, b1, W2, b2 (all float32 LE)
    out = struct.pack("<5i", 0x53484D42, d, D, H, C)
    out += E.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-sharemb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x53484D42), d, D, H, C (int32); then float32
 * E[256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x53484D42) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 20L + 4L * ((long)256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *E = p; p += 256 * D;
    const float *W1 = p; p += (long)d * D * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* concat embeddings */
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * D;
            for (int v = 0; v < D; v++) {
                float ev = e[v];
                const float *w = W1 + ((long)j * D + v) * H;
                for (int u = 0; u < H; u++) h[u] += ev * w[u];
            }
        }
        /* ReLU */
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        /* output layer */
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) logits[c] += hu * w[c];
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```