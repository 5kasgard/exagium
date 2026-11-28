```hypothesis
Family: label-block (block-quantized linear model). The 8x8 digit images have strong local spatial structure (strokes, loops) that 2x2 blocks capture well. Quantizing each pixel to 4 levels (0-3) and learning a 10-class weight vector per block-pattern (16 blocks × 256 patterns) yields ~41K parameters. Trained with Adam like label-linear, it should exploit locality better than global linear (0.9613) at ~1/3 the inference cost (16 table lookups vs 64). Failure mode: if 4-level quantization discards too much ordinal information, accuracy falls below 0.96.
```
```python
"""Block-quantized linear model for 8x8 digit images.
python3 -s -P train.py card.json train.in train.key weights.bin
Quantizes each pixel to 2 bits (0-3), groups 8x8 into 16 non-overlapping 2x2 blocks.
Each block yields an 8-bit pattern index (0-255). Learns a 10-class weight vector per (block, pattern).
Exported as: magic 0x424C4F43, d=64, H=16 (num blocks), C=10; float32 T[16][256][10], bias[10]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x424C4F43  # 'BLOC'
BLOCKS = 16
PATTERNS = 256
Q_LEVELS = 4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def quantize(pixels):
    # pixels: (n, 64) uint8 in 0..16 -> (n, 16) uint8 pattern indices 0..255
    q = np.minimum(pixels // 4, 3).astype(np.uint8)  # 0..3
    n = q.shape[0]
    patterns = np.zeros((n, BLOCKS), dtype=np.uint8)
    # 8x8 row-major, 16 blocks of 2x2 in raster order
    for bi in range(BLOCKS):
        br, bc = divmod(bi, 4)          # block row, col in 0..3
        r0, c0 = br * 2, bc * 2         # top-left pixel
        idx = (r0 * 8 + c0,
               r0 * 8 + c0 + 1,
               (r0 + 1) * 8 + c0,
               (r0 + 1) * 8 + c0 + 1)
        # pack 4 2-bit values into 8 bits: q0 | q1<<2 | q2<<4 | q3<<6
        patterns[:, bi] = (q[:, idx[0]] |
                           (q[:, idx[1]] << 2) |
                           (q[:, idx[2]] << 4) |
                           (q[:, idx[3]] << 6))
    return patterns


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert X.shape[1] == 64 and C == 10

    rng = np.random.default_rng(SEED)
    # weights: (BLOCKS, PATTERNS, C)
    W = (rng.uniform(-1, 1, (BLOCKS, PATTERNS, C)) * math.sqrt(6.0 / PATTERNS)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [W, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    P = quantize(X)  # (n, BLOCKS) uint8
    n = P.shape[0]

    def forward(P_batch):
        # P_batch: (b, BLOCKS) -> logits (b, C)
        logits = np.zeros((len(P_batch), C), np.float32)
        for bi in range(BLOCKS):
            logits += W[bi, P_batch[:, bi]]
        logits += bias
        return logits

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Pb, yy = P[idx], y[idx]
            logits = forward(Pb)
            g = softmax_grad(logits, yy)  # (b, C)
            # grad_W[bi, p, c] += sum_{i: Pb[i,bi]==p} g[i,c]
            grad_W = np.zeros_like(W)
            for bi in range(BLOCKS):
                # scatter-add: for each pattern p, sum g[i] where Pb[i,bi]==p
                # use bincount per class
                for c in range(C):
                    np.add.at(grad_W[bi, :, c], Pb[:, bi], g[:, c])
            grad_W += L2 * W
            grad_bias = g.sum(axis=0) + L2 * bias
            grads = [grad_W, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    out = struct.pack("<4i", MAGIC, 64, BLOCKS, C)
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Block-quantized linear model for 8x8 digits: ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixels 0..16). Output: 1 byte class 0..9.
 * Weights format: magic 0x424C4F43, d=64, H=16 (blocks), C=10;
 * float32 T[16][256][10], bias[10]. */
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
    if (nw < 16 || hd[0] != 0x424C4F43) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], B = hd[2], C = hd[3];
    if (d != 64 || B != 16 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (16L * 256 * 10 + 10);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);           // [16][256][10]
    const float *bias = T + 16L * 256 * 10;             // [10]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float acc[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        for (int c = 0; c < 10; c++) acc[c] = bias[c];

        // 16 blocks in raster order over 8x8
        for (int bi = 0; bi < 16; bi++) {
            int br = bi / 4, bc = bi % 4;
            int r0 = br * 2, c0 = bc * 2;
            // pixel indices in row-major 8x8
            int p0 = r0 * 8 + c0;
            int p1 = p0 + 1;
            int p2 = p0 + 8;
            int p3 = p0 + 9;
            // quantize to 2 bits: val // 4, clamped to 3
            int q0 = xi[p0] >> 2; if (q0 > 3) q0 = 3;
            int q1 = xi[p1] >> 2; if (q1 > 3) q1 = 3;
            int q2 = xi[p2] >> 2; if (q2 > 3) q2 = 3;
            int q3 = xi[p3] >> 2; if (q3 > 3) q3 = 3;
            int pat = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6); // 0..255

            const float *w = T + ((long)bi * 256 + pat) * 10;
            for (int c = 0; c < 10; c++) acc[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```