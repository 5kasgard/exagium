```hypothesis
Family: label-embed (per-position byte embeddings summed, then linear).
Why: The 16-byte context has categorical bytes; a linear model uses 16*256*65 params. Factorized embeddings (E=32) reduce params to ~133k, capture character similarities, and should improve generalization over label-linear (0.3375) while being faster at inference (16*E adds + E*C muls vs 16*256*C).
Failure: If dev accuracy <= 0.3375 or inference cost (instructions/item) exceeds label-linear.
```

```python
import json, math, struct, sys
import numpy as np

E = 32
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
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

    embeddings = []
    for _ in range(d):
        emb = rng.normal(0, math.sqrt(1.0/256), size=(256, E)).astype(np.float32)
        embeddings.append(emb)
    W_out = rng.normal(0, math.sqrt(2.0/(E+C)), size=(E, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = embeddings + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            bsize = len(yb)

            sum_emb = np.zeros((bsize, E), dtype=np.float32)
            for j in range(d):
                sum_emb += embeddings[j][xb[:, j]]

            scores = sum_emb @ W_out + b_out
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            dscores = probs / bsize

            grad_W_out = sum_emb.T @ dscores + L2 * W_out
            grad_b_out = dscores.sum(axis=0)
            grad_sum_emb = dscores @ W_out.T

            grad_embeddings = []
            for j in range(d):
                grad_emb = np.zeros((256, E), dtype=np.float32)
                np.add.at(grad_emb, xb[:, j], grad_sum_emb)
                grad_embeddings.append(grad_emb)

            grads = grad_embeddings + [grad_W_out, grad_b_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x44424D45
    out = struct.pack("<4i", magic, d, E, C)
    for emb in embeddings:
        out += emb.astype("<f4").tobytes()
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
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1] << 8 | (uint32_t)buf[*pos+2] << 16 | (uint32_t)buf[*pos+3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    unsigned char *weights_data = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)weights_data;
    if (hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1];
    int E = hd[2];
    int C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)d * 256 * E * 4 + (long)E * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embeddings = (const float *)(weights_data + 16);
    const float *W_out = embeddings + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *acc = (float *)malloc(sizeof(float) * E);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!acc || !scores) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += emb[e];
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int e = 0; e < E; e++) sum += acc[e] * W_out[(long)e * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```