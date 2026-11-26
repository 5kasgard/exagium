```hypothesis
Family: label-mlp8 — an 8-unit ReLU MLP with per-byte lookup-table inference.
Why: Optdigits pixels exhibit local stroke correlations (corners, loops) that a linear model cannot capture; 8 hidden units provide just enough non-linearity to model these while adding only ~592 FLOPs/item (vs 640 for linear) due to smaller first-layer tables (64×256×8 vs 64×256×10). The fixed He-initialized Adam training is deterministic and fast.
Failure signal: dev accuracy ≤ 0.961 (linear's 0.9613) or Ir/item ≥ 5200 (linear's 5058).
```

```python
import json, math, struct, sys
import numpy as np

H = 8
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
    rng = np.random.default_rng(SEED)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    # He initialization for ReLU
    W1 = (rng.normal(0, math.sqrt(2.0 / k), (k, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh *= (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
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

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void write_rec(FILE *f, const unsigned char *rec, uint32_t len) {
    unsigned char hdr[4] = {len & 255, (len>>8)&255, (len>>16)&255, (len>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, len, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long wsize, isize = 0;
    int32_t *hdr = (int32_t *)slurp(argv[1], &wsize);
    if (wsize < 16 || hdr[0] != 0x54424C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], H = hdr[2], C = hdr[3];
    if (C < 1 || C > 256 || H != 8) { fprintf(stderr, "bad weights: expected H=8\n"); return 3; }
    int W = H;
    long expected = 16 + 4L * ((long)d * 256 * W + W + (long)H * C + C);
    if (wsize != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hdr + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *ibuf = NULL;
    long N = atol(argv[3]);
    if (N > 0) ibuf = slurp(argv[2], &isize);

    float *hidden = (float *)malloc(sizeof(float) * W);
    float *scores = (float *)malloc(sizeof(float) * C);
    long pos = 0;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *x = next_rec(ibuf, isize, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld: %u bytes, expected %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) hidden[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + x[j]) * W;
            for (int w = 0; w < W; w++) hidden[w] += t[w];
        }
        for (int w = 0; w < W; w++) if (hidden[w] < 0.0f) hidden[w] = 0.0f;
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = hidden[u];
            for (int c = 0; c < C; c++) scores[c] += hu * W2[(long)u * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        write_rec(stdout, &best, 1);
    }
    return 0;
}
```