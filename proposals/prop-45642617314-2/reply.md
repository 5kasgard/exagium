```hypothesis
Family: label-embedding (positional embeddings + linear). 
Why: Character-level sequences exhibit strong local similarity; embeddings capture this with far fewer parameters than one-hot linear, improving generalization. 
Failure: Dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item ≥ label-linear.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 32            # embedding dimension
LR = 0.01
BATCH = 64
EPOCHS = 10
L2 = 1e-5
MAGIC = 0x454D4244  # 'DBME' little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    n = len(X_bytes)
    d = card["input"]["bytes"]   # 16
    C = card["classes"]          # 65

    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([b[0] for b in y_bytes], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    # embeddings: [d, 256, E]
    embed = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    W = rng.normal(0, 0.01, size=(E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_embed = np.zeros_like(embed)
    v_embed = np.zeros_like(embed)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # forward: sum embeddings
            emb_sum = np.zeros((B, E), dtype=np.float32)
            for j in range(d):
                emb_sum += embed[j, xb[:, j]]   # (B, E)

            logits = emb_sum @ W + b            # (B, C)

            # softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # gradients
            grad_W = emb_sum.T @ grad_logits + L2 * W       # (E, C)
            grad_b = grad_logits.sum(axis=0)                # (C,)
            grad_emb_sum = grad_logits @ W.T                # (B, E)

            grad_embed = np.zeros_like(embed)
            for j in range(d):
                np.add.at(grad_embed[j], xb[:, j], grad_emb_sum)
            grad_embed += L2 * embed

            # Adam update
            t += 1
            # embed
            m_embed = beta1 * m_embed + (1 - beta1) * grad_embed
            v_embed = beta2 * v_embed + (1 - beta2) * (grad_embed * grad_embed)
            m_hat = m_embed / (1 - beta1 ** t)
            v_hat = v_embed / (1 - beta2 ** t)
            embed -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
            m_hat = m_b / (1 - beta1 ** t)
            v_hat = v_b / (1 - beta2 ** t)
            b -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, d, E, C)
    out += embed.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: positional embeddings + linear.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 16 || hd[0] != 0x454D4244) {  // magic
        fprintf(stderr, "bad weights\n");
        return 3;
    }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) {
        fprintf(stderr, "bad header\n");
        return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) {
        fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected);
        return 3;
    }
    const float *embed = (const float *)(hd + 4);
    const float *W = embed + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *xbuf = N ? slurp(argv[2], &nx) : NULL;

    float *acc = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(xbuf, nx, &pos, &k);
        if ((int)k != d) {
            fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d);
            return 3;
        }
        // zero accumulator
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        // sum embeddings
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += e[e];
        }
        // scores = acc @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int e = 0; e < E; e++) {
            float a = acc[e];
            if (a != 0.0f) {
                const float *w = W + (long)e * C;
                for (int c = 0; c < C; c++) scores[c] += a * w[c];
            }
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```