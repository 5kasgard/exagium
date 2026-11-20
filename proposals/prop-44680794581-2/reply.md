```hypothesis
Family: label-rnn (simple recurrent network). Processes the 16-byte context sequentially with a 32-dim embedding and 32-dim hidden state (tanh), capturing sequential dependencies that position-wise linear models miss. Trained with Adam (lr=1e-3, L2=1e-4) via BPTT on minibatches. Fails if dev accuracy ≤ 0.3375 (label-linear) showing sequence structure isn't exploitable at this capacity.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
EMBED = 32
HIDDEN = 32
LR = 1e-3
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    embed = (rng.normal(0, 1 / math.sqrt(EMBED), (256, EMBED))).astype(np.float32)
    W_xh = (rng.normal(0, 1 / math.sqrt(EMBED), (EMBED, HIDDEN))).astype(np.float32)
    W_hh = (rng.normal(0, 1 / math.sqrt(HIDDEN), (HIDDEN, HIDDEN))).astype(np.float32)
    b_h = np.zeros(HIDDEN, np.float32)
    W_out = (rng.normal(0, 1 / math.sqrt(HIDDEN), (HIDDEN, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [embed, W_xh, W_hh, b_h, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            hs = np.zeros((B, d + 1, HIDDEN), np.float32)
            for step in range(d):
                x_emb = embed[xb[:, step]]
                hs[:, step + 1] = np.tanh(x_emb @ W_xh + hs[:, step] @ W_hh + b_h)
            logits = hs[:, -1] @ W_out + b_out

            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            grad_W_out = hs[:, -1].T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(0)
            grad_h = grad_logits @ W_out.T

            grad_W_xh = np.zeros_like(W_xh)
            grad_W_hh = np.zeros_like(W_hh)
            grad_b_h = np.zeros_like(b_h)
            grad_embed = np.zeros_like(embed)

            for step in reversed(range(d)):
                tanh_grad = 1 - hs[:, step + 1] ** 2
                grad_pre = grad_h * tanh_grad

                grad_W_xh += embed[xb[:, step]].T @ grad_pre
                grad_W_hh += hs[:, step].T @ grad_pre
                grad_b_h += grad_pre.sum(0)
                np.add.at(grad_embed, xb[:, step], grad_pre @ W_xh.T)

                grad_h = grad_pre @ W_hh.T

            grad_W_xh += L2 * W_xh
            grad_W_hh += L2 * W_hh
            grad_embed += L2 * embed

            grads = [grad_embed, grad_W_xh, grad_W_hh, grad_b_h, grad_W_out, grad_b_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    MAGIC = 0x524E4E01
    out = struct.pack("<5i", MAGIC, d, EMBED, HIDDEN, C)
    out += embed.astype("<f4").tobytes()
    out += W_xh.astype("<f4").tobytes()
    out += W_hh.astype("<f4").tobytes()
    out += b_h.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x524E4E01) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * (256L * E + (long)E * H + (long)H * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *embed = p; p += 256 * E;
    const float *W_xh = p; p += E * H;
    const float *W_hh = p; p += H * H;
    const float *b_h = p; p += H;
    const float *W_out = p; p += H * C;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(H * sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *x_emb = malloc(E * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, H * sizeof(float));
        for (int t = 0; t < d; t++) {
            const float *e = embed + xi[t] * E;
            memcpy(x_emb, e, E * sizeof(float));
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k2 = 0; k2 < E; k2++) sum += x_emb[k2] * W_xh[k2 * H + j];
                for (int k2 = 0; k2 < H; k2++) sum += h[k2] * W_hh[k2 * H + j];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }

        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_out[j * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(h_new); free(x_emb); free(logits);
    return 0;
}
```