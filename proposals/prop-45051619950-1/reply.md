```hypothesis
Family: label-cnn1 (tiny CNN: 3x3 conv, 8 filters, ReLU, global average pooling, linear).
Why: Exploits 2D spatial locality in 8x8 digit images; weight sharing reduces parameters vs MLP while capturing stroke features. Per-pixel z-score normalization matches label-linear preprocessing.
Failure signal: dev accuracy ≤ 0.961 (label-linear) or instructions/item ≥ 5059 (label-linear).
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CONV_OUT = 8
KERNEL = 3

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    assert d == 64, "expected 64 bytes"
    H = W = 8

    Xf = X.astype(np.float64)
    pixel_mean = Xf.mean(axis=0)
    pixel_std = Xf.std(axis=0)
    pixel_std = np.maximum(pixel_std, 1e-8)
    Xn = ((Xf - pixel_mean) / pixel_std).astype(np.float32)
    Xn = Xn.reshape(n, 1, H, W)

    rng = np.random.default_rng(SEED)

    fan_in = 1 * KERNEL * KERNEL
    conv_w = (rng.normal(0, math.sqrt(2.0 / fan_in), (CONV_OUT, 1, KERNEL, KERNEL))).astype(np.float32)
    conv_b = np.zeros(CONV_OUT, np.float32)
    linear_w = (rng.uniform(-1, 1, (CONV_OUT, C)) * math.sqrt(6.0 / (CONV_OUT + C))).astype(np.float32)
    linear_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, linear_w, linear_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = Xn[idx]
            y_batch = y[idx]
            B = len(idx)

            padded = np.pad(x_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
            patches = np.lib.stride_tricks.sliding_window_view(padded, (KERNEL, KERNEL), axis=(2,3))
            patches = patches.reshape(B, 64, 9)

            conv_w_flat = conv_w.reshape(CONV_OUT, 9)
            conv_out = patches @ conv_w_flat.T + conv_b
            conv_out = conv_out.transpose(0,2,1).reshape(B, CONV_OUT, H, W)

            h = np.maximum(conv_out, 0)
            g = h.mean(axis=(2,3))

            logits = g @ linear_w + linear_b
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            dlogits = probs
            dlogits[np.arange(B), y_batch] -= 1.0
            dlogits /= B

            dlinear_w = g.T @ dlogits + L2 * linear_w
            dlinear_b = dlogits.sum(axis=0)

            dg = dlogits @ linear_w.T
            dh = dg[:, :, None, None] / 64.0
            dconv_out = dh * (conv_out > 0)

            dconv_out_flat = dconv_out.transpose(0,2,3,1).reshape(B, 64, CONV_OUT)
            dconv_w_flat = np.sum(np.matmul(patches.transpose(0,2,1), dconv_out_flat), axis=0).T
            dconv_w = dconv_w_flat.reshape(CONV_OUT, 1, KERNEL, KERNEL) + L2 * conv_w
            dconv_b = dconv_out_flat.sum(axis=(0,1))

            grads = [dconv_w, dconv_b, dlinear_w, dlinear_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x434E4E31
    out = struct.pack("<5i", magic, d, C, CONV_OUT, KERNEL)
    out += pixel_mean.astype("<f4").tobytes()
    out += pixel_std.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += linear_w.astype("<f4").tobytes()
    out += linear_b.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], conv_out = hd[3], kernel = hd[4];
    if (d != 64 || C != 10 || conv_out != 8 || kernel != 3) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    long expected_floats = 64 + 64 + conv_out * 1 * kernel * kernel + conv_out + conv_out * C + C;
    if (nw != 20 + expected_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(hd + 5);
    float *pixel_mean = ptr; ptr += 64;
    float *pixel_std = ptr; ptr += 64;
    float *conv_w = ptr; ptr += conv_out * 1 * kernel * kernel;
    float *conv_b = ptr; ptr += conv_out;
    float *linear_w = ptr; ptr += conv_out * C;
    float *linear_b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float xn[64];
        for (int j = 0; j < 64; j++) {
            xn[j] = (xi[j] - pixel_mean[j]) / pixel_std[j];
        }

        float h[8][8][8];
        for (int f = 0; f < conv_out; f++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    float sum = conv_b[f];
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ii = i + di;
                            int jj = j + dj;
                            if (ii >= 0 && ii < 8 && jj >= 0 && jj < 8) {
                                int w_idx = f * 9 + (di + 1) * 3 + (dj + 1);
                                sum += xn[ii * 8 + jj] * conv_w[w_idx];
                            }
                        }
                    }
                    h[f][i][j] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        float g[8];
        for (int f = 0; f < conv_out; f++) {
            float sum = 0.0f;
            for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) sum += h[f][i][j];
            g[f] = sum / 64.0f;
        }

        float scores[10];
        for (int c = 0; c < C; c++) {
            scores[c] = linear_b[c];
            for (int f = 0; f < conv_out; f++) {
                scores[c] += g[f] * linear_w[f * C + c];
            }
        }

        int best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;

        unsigned char out_byte = (unsigned char)best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```