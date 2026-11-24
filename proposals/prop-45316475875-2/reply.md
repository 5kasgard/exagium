```hypothesis
Family: label-embed (factorized position–character embeddings + linear head).
Why: label-linear uses 16×256 one-hot (4096 dims) but only 65 chars occur → no sharing, high variance.
Factorized embeddings (16 positions × 66 vocab × 32 dims) share stats across positions/chars,
giving ~66K params vs 266K, better generalization at similar inference cost (512 features → 33K MACs).
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference > 10k Ir/item.
```

```python
import json, math, struct, sys

import numpy as np

EMBED_DIM = 32
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
    C = card["classes"]          # 65
    d = card["input"]["bytes"]   # 16

    # byte value -> class index (0..64), unknown -> 65
    byte_to_class = np.full(256, C, dtype=np.int64)
    for idx, ch in enumerate(card["class_chars"]):
        byte_to_class[ord(ch)] = idx
    X_cls = byte_to_class[X]     # (n, 16)

    rng = np.random.default_rng(SEED)
    vocab = C + 1                # 66 including unknown
    E = EMBED_DIM

    # embeddings: (d, vocab, E)
    emb = rng.uniform(-0.1, 0.1, (d, vocab, E)).astype(np.float32)
    # linear: (d*E, C)
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_cls = X_cls[idx]   # (B, 16)
            yy = y[idx]

            # gather embeddings
            emb_batch = emb[np.arange(d)[None, :], x_cls]      # (B, 16, E)
            z = emb_batch.reshape(len(idx), -1)                # (B, 16*E)

            # linear logits
            logits = z @ W + b                                 # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)           # (B, C)

            # gradients
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grad_z = g @ W.T                                   # (B, 16*E)
            grad_emb_batch = grad_z.reshape(len(idx), d, E)    # (B, 16, E)

            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, (np.arange(d)[None, :], x_cls), grad_emb_batch)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # save: magic, d, E, C; then emb[d][vocab][E], W[d*E][C], b[C]
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int vocab = C + 1;
    long expected = 16 + 4L * ((long)d * vocab * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * vocab * E;
    const float *b = W + (long)d * E * C;

    /* byte -> class index (0..64), unknown -> 65 */
    unsigned char byte_to_class[256];
    for (int i = 0; i < 256; i++) byte_to_class[i] = C;
    const char *class_chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < C; i++) byte_to_class[(unsigned char)class_chars[i]] = i;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            int cls = byte_to_class[xi[j]];
            const float *src = emb + ((long)j * vocab + cls) * E;
            float *dst = z + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }

        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = b[c];
            for (int fe = 0; fe < d * E; fe++)
                score += z[fe] * W[(long)fe * C + c];
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```