```hypothesis
Family: label-embed (per-position byte embeddings + linear classifier).
Why: The 16-byte context consists of characters with semantic similarity (e.g., 'a'/'b', 'A'/'a'). One-hot linear (label-linear) ignores this; embeddings capture it with far fewer parameters (16*256*E + 16*E*65 vs 4096*65). A small embedding size (E=16) keeps inference fast (256-D dot product) while potentially improving generalization over the 0.3375 linear baseline.
Failure: If dev accuracy ≤ 0.3375 or CPU cost per item exceeds label-linear, the family does not justify its complexity.
```

```python
"""label-embed: per-position byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D4245 ('EMBE'), d=16, E, C=65
  float32 emb[d][256][E]       (d*256*E)
  float32 W_out[d*E][C]        (d*E*C)
  float32 b_out[C]             (C)
"""
import json, math, struct, sys

import numpy as np

E = 16                     # embedding dimension per position
SEED = 1
LR = 3e-3
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: [d, 256, E]
    scale = math.sqrt(6.0 / (256 + E))
    emb = (rng.uniform(-1, 1, (d, 256, E)) * scale).astype(np.float32)
    # output layer: [d*E, C]
    D = d * E
    scale2 = math.sqrt(6.0 / (D + C))
    W_out = (rng.uniform(-1, 1, (D, C)) * scale2).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: gather embeddings
            # emb_gathered shape [bsz, d, E] -> reshape to [bsz, D]
            emb_gathered = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2).reshape(bsz, D)
            logits = emb_gathered @ W_out + b_out            # [bsz, C]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient w.r.t logits
            grad_logits = probs
            grad_logits[np.arange(bsz), yb] -= 1.0
            grad_logits /= bsz

            # grad_W_out = emb_gathered.T @ grad_logits  [D, C]
            grad_W_out = emb_gathered.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)

            # grad_emb_gathered = grad_logits @ W_out.T  [bsz, D]
            grad_emb_gathered = grad_logits @ W_out.T
            grad_emb_gathered = grad_emb_gathered.reshape(bsz, d, E).transpose(1, 0, 2)  # [d, bsz, E]

            # accumulate into emb gradients
            grad_emb = np.zeros_like(emb)
            for p in range(d):
                np.add.at(grad_emb[p], xb[:, p], grad_emb_gathered[p])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W_out, grad_b_out]

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D4245, d, E, C
 *   float32 emb[d][256][E]
 *   float32 W_out[d*E][C]
 *   float32 b_out[C]
 */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    int D = d * E;
    long expected = 16L + 4L * ((long)d * 256 * E + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // gather embeddings
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * E;
            memcpy(concat + p * E, e, sizeof(float) * E);
        }

        // scores = concat @ W_out + b_out
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int j = 0; j < D; j++) {
            float v = concat[j];
            if (v == 0.0f) continue;
            const float *w = W_out + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += v * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```