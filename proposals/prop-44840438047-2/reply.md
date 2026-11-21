```hypothesis
Family: label-emb (shared character embeddings + position embeddings, linear head).
Why: The 16-byte context consists of characters from a small alphabet (65 classes). A linear model on one-hot per position (label-linear) has 16*256*65 ≈ 266k parameters and ignores that the same character appears in different positions. Shared character embeddings (256×D) plus position embeddings (16×D) reduce parameters to ~10k (D=8), forcing generalization across positions and reducing overfitting on 24k examples. The linear head on concatenated (char_emb + pos_emb) preserves position sensitivity. If this does not beat label-linear (0.3375), it indicates position-specific one-hot features are essential for this task.
```

```python
"""label-emb: shared character embeddings + position embeddings, linear head.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D424C, d, D, C; float32 char_emb[256][D], pos_emb[d][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 8
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

    # Parameters
    char_emb = (rng.normal(0, 1/math.sqrt(D), (256, D))).astype(np.float32)
    pos_emb = (rng.normal(0, 1/math.sqrt(D), (d, D))).astype(np.float32)
    W = (rng.normal(0, 1/math.sqrt(d*D), (d*D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [char_emb, pos_emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]  # (B, d)
            yy = y[idx]  # (B,)
            B = xi.shape[0]

            # Forward: build concatenated embeddings (B, d*D)
            # char_emb[xi] -> (B, d, D), add pos_emb -> (B, d, D), reshape
            emb = char_emb[xi] + pos_emb  # broadcasting (B,d,D) + (d,D)
            emb_flat = emb.reshape(B, d*D)  # (B, d*D)

            # Scores
            scores = emb_flat @ W + b  # (B, C)
            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_scores = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            grad_W = emb_flat.T @ grad_scores + L2 * W  # (d*D, C)
            grad_b = grad_scores.sum(axis=0)  # (C,)
            grad_emb_flat = grad_scores @ W.T  # (B, d*D)
            grad_emb = grad_emb_flat.reshape(B, d, D)  # (B, d, D)

            # Gradient for char_emb: scatter-add over batch and positions
            grad_char_emb = np.zeros_like(char_emb)
            # xi is (B, d) indices, grad_emb is (B, d, D)
            # We can loop over positions (d=16) for simplicity
            for j in range(d):
                np.add.at(grad_char_emb, xi[:, j], grad_emb[:, j, :])
            grad_char_emb += L2 * char_emb

            # Gradient for pos_emb: sum over batch
            grad_pos_emb = grad_emb.sum(axis=0) + L2 * pos_emb  # (d, D)

            grads = [grad_char_emb, grad_pos_emb, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424C  # "EMBL"
    out = struct.pack("<4i", magic, d, D, C)
    out += char_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D424C, int32 d, D, C; float32 char_emb[256][D], pos_emb[d][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * (256L * D + (long)d * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *char_emb = (const float *)(hd + 4);
    const float *pos_emb = char_emb + 256L * D;
    const float *W = pos_emb + (long)d * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_flat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embeddings
        for (int j = 0; j < d; j++) {
            const float *ce = char_emb + (long)xi[j] * D;
            const float *pe = pos_emb + (long)j * D;
            float *out = emb_flat + (long)j * D;
            for (int u = 0; u < D; u++) out[u] = ce[u] + pe[u];
        }

        // Scores = emb_flat @ W + b
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < d * D; u++) sum += emb_flat[u] * W[(long)u * C + c];
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```