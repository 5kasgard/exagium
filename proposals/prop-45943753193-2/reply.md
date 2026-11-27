```hypothesis
Family: label-sharedemb (shared byte embeddings + position-specific linear heads).
Why: The linear model learns independent weights for each (position, byte) pair (16*256*65 params). A shared embedding of size E=16 reduces parameters to ~20k while allowing the model to learn byte similarities (e.g., 'a' and 'b' have similar embeddings) and position-specific projections. This should improve generalization on the character-prediction task where byte semantics are shared across positions. If dev accuracy does not exceed 0.34 (label-linear) or inference cost exceeds 2x label-linear, the hypothesis fails.
```

```python
"""label-sharedemb: shared byte embeddings (E=16) with position-specific linear heads.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only. Exports: magic, d, E, C, emb[256][E], W[d][E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 16
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embedding matrix: 256 x E
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # Position-specific weights: d x E x C
    W = (rng.normal(0, math.sqrt(2.0 / E), (d, E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            logits = np.zeros((bs, C), np.float32)
            for j in range(d):
                e = emb[xb[:, j]]                 # (bs, E)
                logits += e @ W[j]                # (bs, C)
            logits += b

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            grad_b = probs.sum(axis=0).astype(np.float32)
            grad_W = np.zeros_like(W)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                e = emb[xb[:, j]]                 # (bs, E)
                grad_W[j] = e.T @ probs + L2 * W[j]
                # scatter-add into grad_emb
                g_e = probs @ W[j].T              # (bs, E)
                np.add.at(grad_emb, xb[:, j], g_e)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    magic = 0x4D454853  # 'SHEM' little-endian
    header = struct.pack("<4i", magic, d, E, C)
    out = header + emb.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-sharedemb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x4D454853) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * (256L * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + 256L * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *tmp = malloc(sizeof(float) * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            const float *wj = W + (long)j * E * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * wj[(long)ee * C + c];
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