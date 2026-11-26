```hypothesis
Family: label-embed (positional embeddings + linear). Each of the 16 context positions gets a learned 256×E embedding table; the 16 E-d vectors are concatenated and fed to a linear 65-class head. This factorizes the one-hot linear model (4096→65) into 16×256×E + 16E×65 parameters, creating a bottleneck that should regularize better on 24k samples while keeping inference extremely fast (16×E adds + one matvec). If dev accuracy ≤ 0.3375 (label-linear) the bottleneck loses too much position-specific information for this task.
```
```python
"""label-embed: positional embeddings + linear head for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: int32 magic 0x454D4245, d, E, C; float32 emb[d][256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # [B, d]
            yy = y[idx]          # [B]

            # forward: accumulate embeddings
            B = xi.shape[0]
            h = np.zeros((B, d * E), np.float32)
            for j in range(d):
                h[:, j*E:(j+1)*E] = emb[j, xi[:, j], :]
            logits = h @ W + b

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            probs /= B

            # gradients
            grad_W = h.T @ probs + L2 * W
            grad_b = probs.sum(axis=0)
            grad_h = probs @ W.T

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # grad_h[:, j*E:(j+1)*E] accumulates into grad_emb[j, xi[:, j], :]
                gj = grad_h[:, j*E:(j+1)*E]
                np.add.at(grad_emb[j], xi[:, j], gj)

            grads = [grad_emb, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    out = struct.pack("<4i", 0x454D4245, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed: positional embeddings + linear head. ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (argmax, first max wins).
 * Weights format: int32 magic 0x454D4245, d, E, C; float32 emb[d][256][E], W[d*E][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long emb_sz = (long)d * 256 * E;
    long W_sz = (long)d * E * C;
    long expect = 16 + 4L * (emb_sz + W_sz + C);
    if (nw != expect) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * d * E);
    float *s = (float *)malloc(sizeof(float) * C);
    if (!h || !s) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = concat of embeddings + bias later
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            float *hj = h + j * E;
            for (int u = 0; u < E; u++) hj[u] = e[u];
        }

        // logits = h @ W + b
        for (int c = 0; c < C; c++) {
            float acc = b[c];
            for (int u = 0; u < d * E; u++) acc += h[u] * W[(long)u * C + c];
            s[c] = acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```