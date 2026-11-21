```hypothesis
Family: label-embed (position-specific byte embeddings + linear classifier).
Why: Character-level LM benefits from learned byte similarities (e.g., 'a'≈'b', digits cluster); 16 position-specific 32-d embeddings (512 features) generalize better than 4096 one-hots while being 8× smaller and faster. Train with Adam on full train set.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ label-linear.
```
```python
"""label-embed: position-specific byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: magic 0x454D4244, d=32, d_pos=16, C=65; float32 E[16][256][d], W[16*d][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4244  # "EMBD" little-endian


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
    d_pos = card["input"]["bytes"]
    C = card["classes"]
    assert d_pos == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # Embeddings: [d_pos, 256, D]
    E = (rng.normal(0, 0.01, (d_pos, 256, D))).astype(np.float32)
    # Output weights: [d_pos*D, C], bias: [C]
    W = (rng.normal(0, 0.01, (d_pos * D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bsz = xi.shape[0]

            # Forward: embed and concat
            # xi: [bsz, 16] -> gather E[16, xi, :] -> [bsz, 16, D] -> [bsz, 16*D]
            emb = E[np.arange(d_pos)[:, None], xi.T].transpose(1, 0, 2).reshape(bsz, -1)  # [bsz, 512]
            logits = emb @ W + b  # [bsz, C]

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)  # [bsz, C]

            # Gradients
            grad_W = emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb = grad_logits @ W.T  # [bsz, 512]
            grad_emb = grad_emb.reshape(bsz, d_pos, D)
            grad_E = np.zeros_like(E)
            for j in range(d_pos):
                np.add.at(grad_E[j], xi[:, j], grad_emb[:, j])

            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # Magic, d, d_pos, C
    header = struct.pack("<4i", MAGIC, D, d_pos, C)
    out = header + E.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed: position-specific byte embeddings + linear classifier.
 * ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index 0..64). */
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
    int D = hd[1], d_pos = hd[2], C = hd[3];
    if (d_pos != 16 || C != 65 || D <= 0 || D > 128 || C > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d_pos * 256 * D + (long)d_pos * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d_pos * 256 * D;
    const float *b = W + (long)d_pos * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d_pos * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_pos) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_pos); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d_pos; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            memcpy(emb + (long)j * D, e, D * sizeof(float));
        }

        // Logits = emb @ W + b
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int f = 0; f < d_pos * D; f++) {
            float ef = emb[f];
            if (ef != 0.0f) {
                const float *w = W + (long)f * C;
                for (int c = 0; c < C; c++) logits[c] += ef * w[c];
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```