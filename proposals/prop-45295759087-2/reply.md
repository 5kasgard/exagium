```hypothesis
Shared embedding + positional encoding model (family: label-embedpos). Learns a 64-dim embedding per byte value (256) and per position (16), averages the 16 summed embeddings, then linear to 65 classes. Captures character identity invariance across positions (e.g., 't' similar at pos 0 and 15) while retaining position info via learned pos_emb. ~21K params, fast inference (16*E lookups + E*C matvec). Beats label-linear by sharing statistical strength across positions. Fails if task needs explicit n-gram interactions beyond positional averaging.
```
```python
import json, math, struct, sys

import numpy as np

E = 64
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 30


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    pos_emb = (rng.normal(0, 0.01, (d, E))).astype(np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / E), (E, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [emb, pos_emb, W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = x.shape[0]

            emb_x = emb[x]
            h = emb_x + pos_emb[None, :, :]
            h = h.mean(axis=1)
            logits = h @ W_out + b_out

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = probs / B

            grad_W_out = h.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W_out.T
            grad_h /= d

            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, x, grad_h[:, None, :])
            grad_emb += L2 * emb

            grad_pos_emb = grad_h[:, None, :].sum(axis=0) + L2 * pos_emb

            grads = [grad_emb, grad_pos_emb, grad_W_out, grad_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    header = struct.pack("<4i", 0x454D4244, d, E, C)
    body = (emb.astype("<f4").tobytes() +
            pos_emb.astype("<f4").tobytes() +
            W_out.astype("<f4").tobytes() +
            b_out.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * E + d * E + E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *pos_emb = emb + 256L * E;
    const float *W_out = pos_emb + d * E;
    const float *b_out = W_out + E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(E * sizeof(float));
    float *logits = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + (long)xi[j] * E;
            const float *pptr = pos_emb + (long)j * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e] + pptr[e];
        }
        float inv_d = 1.0f / d;
        for (int e = 0; e < E; e++) h[e] *= inv_d;

        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int e = 0; e < E; e++) {
            float he = h[e];
            const float *wptr = W_out + (long)e * C;
            for (int c = 0; c < C; c++) logits[c] += he * wptr[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```