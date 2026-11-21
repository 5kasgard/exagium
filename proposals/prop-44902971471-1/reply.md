```hypothesis
Family: label-spatlin (spatial linear). Adds fixed spatial features (horizontal/vertical gradients and 2x2 block sums) to the raw pixels, then trains a linear model. Captures local stroke structure critical for 8x8 digits without learned hidden layers. Should beat label-linear (0.9613) at modest cost; fails if dev accuracy < 0.965 or Ir/item > 15000.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_features(X):
    n = X.shape[0]
    feat = np.zeros((n, 192), dtype=np.float32)
    feat[:, :64] = X.astype(np.float32)
    idx = 64
    for r in range(8):
        for c in range(7):
            feat[:, idx] = X[:, r * 8 + c + 1].astype(np.float32) - X[:, r * 8 + c].astype(np.float32)
            idx += 1
    for r in range(7):
        for c in range(8):
            feat[:, idx] = X[:, (r + 1) * 8 + c].astype(np.float32) - X[:, r * 8 + c].astype(np.float32)
            idx += 1
    for br in range(4):
        for bc in range(4):
            s = np.zeros(n, dtype=np.float32)
            for dr in range(2):
                for dc in range(2):
                    s += X[:, (2 * br + dr) * 8 + (2 * bc + dc)].astype(np.float32)
            feat[:, idx] = s
            idx += 1
    assert idx == 192
    return feat


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert C == 10 and X.shape[1] == 64

    rng = np.random.default_rng(SEED)
    Z = compute_features(X)
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std[std == 0] = 1.0
    Z = (Z - mean) / std
    n, k = Z.shape
    assert k == 192

    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            g = softmax_grad(S, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    inv_std = 1.0 / std
    out = struct.pack("<4i", 0x53504C31, 64, 192, C)
    out += mean.astype("<f4").tobytes()
    out += inv_std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x53504C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], nfeat = hd[2], C = hd[3];
    if (d != 64 || nfeat != 192 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (nfeat * 2 + nfeat * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *inv_std = mean + nfeat;
    const float *W = inv_std + nfeat;
    const float *b = W + (long)nfeat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * nfeat);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int idx = 0;
        for (int j = 0; j < 64; j++) feat[idx++] = (float)xi[j];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 7; c++)
                feat[idx++] = (float)xi[r * 8 + c + 1] - (float)xi[r * 8 + c];
        for (int r = 0; r < 7; r++)
            for (int c = 0; c < 8; c++)
                feat[idx++] = (float)xi[(r + 1) * 8 + c] - (float)xi[r * 8 + c];
        for (int br = 0; br < 4; br++)
            for (int bc = 0; bc < 4; bc++) {
                float sum = 0.0f;
                for (int dr = 0; dr < 2; dr++)
                    for (int dc = 0; dc < 2; dc++)
                        sum += (float)xi[(2 * br + dr) * 8 + (2 * bc + dc)];
                feat[idx++] = sum;
            }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < nfeat; f++) {
            float val = (feat[f] - mean[f]) * inv_std[f];
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += val * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```