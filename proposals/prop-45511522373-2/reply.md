```hypothesis
label-conv1d: 1D convolution over learned byte embeddings with positional encodings.
Captures local character n-grams (crucial for next-char prediction) with far fewer parameters
and lower inference cost than label-linear. Failure: dev accuracy ≤ 0.3375 or Ir/item ≥ 7359.
```
```python
import json, math, struct, sys
import numpy as np

E, H, K = 32, 64, 3
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

    byte_emb = rng.normal(0, 0.01, (256, E)).astype(np.float32)
    pos_emb = rng.normal(0, 0.01, (d, E)).astype(np.float32)
    conv_w = rng.normal(0, math.sqrt(2.0 / (K * E)), (K, E, H)).astype(np.float32)
    conv_b = np.zeros(H, np.float32)
    fc_w = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [byte_emb, pos_emb, conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    L = d - K + 1
    total_steps = min(MAX_EPOCHS * ((n + BATCH - 1) // BATCH), math.ceil(SAMPLES / BATCH))
    t = 0

    for _ in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            if t >= total_steps:
                break
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            # Forward
            emb = byte_emb[xb] + pos_emb[None, :, :]          # (B, d, E)
            conv_out = np.zeros((B, L, H), np.float32)
            for i in range(L):
                window = emb[:, i:i + K, :]                   # (B, K, E)
                conv_out[:, i, :] = np.einsum('bke,keh->bh', window, conv_w) + conv_b
            conv_out = np.maximum(conv_out, 0)                # ReLU
            pooled = conv_out.max(axis=1)                     # (B, H)
            logits = pooled @ fc_w + fc_b                     # (B, C)

            # Loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            g_logits = probs / B                              # (B, C)

            # Backward
            g_fc_w = pooled.T @ g_logits
            g_fc_b = g_logits.sum(axis=0)
            g_pooled = g_logits @ fc_w.T                      # (B, H)

            g_conv_out = g_pooled[:, None, :] * (conv_out > 0)  # (B, L, H)

            g_conv_w = np.zeros_like(conv_w)
            g_conv_b = g_conv_out.sum(axis=(0, 1))
            g_emb = np.zeros_like(emb)

            for i in range(L):
                window = emb[:, i:i + K, :]
                g_conv_w += np.einsum('bke,bh->keh', window, g_conv_out[:, i, :])
                g_emb[:, i:i + K, :] += np.einsum('bh,keh->bke', g_conv_out[:, i, :], conv_w)

            g_byte_emb = np.zeros_like(byte_emb)
            g_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                np.add.at(g_byte_emb, xb[:, j], g_emb[:, j, :])
                g_pos_emb[j] += g_emb[:, j, :].sum(axis=0)

            grads = [g_byte_emb, g_pos_emb, g_conv_w, g_conv_b, g_fc_w, g_fc_b]
            for p, gr in zip(params, grads):
                gr += L2 * p

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save: magic, d, E, H, K, C
    magic = 0x434F4E56
    out = struct.pack("<6i", magic, d, E, H, K, C)
    out += byte_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += fc_w.astype("<f4").tobytes()
    out += fc_b.astype("<f4").tobytes()
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
    if (nw < 24 || hd[0] != 0x434F4E56) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], K = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights config\n"); return 3; }

    long expected = 24
        + 4L * 256 * E
        + 4L * d * E
        + 4L * K * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 6);
    const float *byte_emb = p; p += 256 * E;
    const float *pos_emb = p; p += d * E;
    const float *conv_w = p; p += K * E * H;
    const float *conv_b = p; p += H;
    const float *fc_w = p; p += H * C;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb = malloc(sizeof(float) * d * E);
    float *conv_out = malloc(sizeof(float) * (d - K + 1) * H);
    float *pooled = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    int L = d - K + 1;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + xi[j] * E;
            const float *pe = pos_emb + j * E;
            float *out = emb + j * E;
            for (int e = 0; e < E; e++) out[e] = be[e] + pe[e];
        }

        for (int l = 0; l < L; l++) {
            float *out = conv_out + l * H;
            for (int h = 0; h < H; h++) out[h] = conv_b[h];
            for (int kk = 0; kk < K; kk++) {
                const float *in = emb + (l + kk) * E;
                const float *w = conv_w + (kk * E * H);
                for (int e = 0; e < E; e++) {
                    float ival = in[e];
                    if (ival == 0.0f) continue;
                    const float *we = w + e * H;
                    for (int h = 0; h < H; h++) out[h] += ival * we[h];
                }
            }
            for (int h = 0; h < H; h++) if (out[h] < 0.0f) out[h] = 0.0f;
        }

        for (int h = 0; h < H; h++) {
            float maxv = conv_out[h];
            for (int l = 1; l < L; l++) {
                float v = conv_out[l * H + h];
                if (v > maxv) maxv = v;
            }
            pooled[h] = maxv;
        }

        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int h = 0; h < H; h++) sum += pooled[h] * fc_w[h * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```