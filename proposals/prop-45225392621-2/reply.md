```hypothesis
Family: label-embed — low-rank factorization of the position×byte×class weight tensor via byte embeddings (E=32) and position-specific projections. The 16-byte context is categorical; factorization shares statistical strength across bytes and positions, regularizing the 266K-parameter linear model down to ~42K parameters. Expect higher dev accuracy than label-linear (0.3375) at ~8× lower inference cost (33K vs 266K FMAs/item). Failure: accuracy ≤0.3375 or Ir/item ≥7000.
```
```python
"""label-embed: low-rank factorized linear model for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Learns byte embeddings (256×E) and position-specific projections (d×E×C) with bias (C).
Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000 / n)). Deterministic seed 1.
Exports: magic 0x454D4231, d, E, C; float32 emb[256][E], W_pos[d][E][C], b[C]."""
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

    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    W_pos = (rng.normal(0, 0.01, (d, E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_pos), np.zeros_like(W_pos)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yy = X[idx], y[idx]
            B = len(yy)

            logits = np.broadcast_to(b, (B, C)).copy()
            for j in range(d):
                logits += emb[xb[:, j]] @ W_pos[j]

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            g_logits = (probs / B).astype(np.float32)

            g_b = g_logits.sum(axis=0) + L2 * b
            g_W = np.zeros_like(W_pos)
            g_emb = np.zeros_like(emb)

            for j in range(d):
                bytes_j = xb[:, j]
                grad_emb_j = g_logits @ W_pos[j].T
                np.add.at(g_emb, bytes_j, grad_emb_j)
                g_W[j] = emb[bytes_j].T @ g_logits + L2 * W_pos[j]
            g_emb += L2 * emb

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            W_pos -= LR * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * g_b
            v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
            b -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4i", 0x454D4231, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W_pos.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (argmax).
 * Weights: magic 0x454D4231, d, E, C; float32 emb[256*E], W_pos[d*E*C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_pos = emb + 256L * E;
    const float *b = W_pos + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b[c];

        for (int j = 0; j < d; j++) {
            unsigned char byte_val = xi[j];
            const float *e = emb + (long)byte_val * E;
            const float *Wj = W_pos + (long)j * E * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int kk = 0; kk < E; kk++) sum += e[kk] * Wj[(long)kk * C + c];
                logits[c] += sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```