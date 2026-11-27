```hypothesis
Family: label-emblinear (positional embeddings + linear output). Each of the 16 context positions has a 65-class embedding table (8-dim). Sum embeddings, then linear layer to 65 classes. Low-rank factorization of the 16*65 x 65 weight matrix reduces params from ~68K to ~9K, improving generalization on small data while keeping inference fast (~650 FLOPs/item). Fails if 8-dim is too small to capture position-char interactions (dev accuracy < 0.33).
```
```python
import json, math, struct, sys
import numpy as np

D = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C, d = card["classes"], card["input"]["bytes"]

    byte_to_class = np.full(256, -1, dtype=np.int8)
    for idx, ch in enumerate(card["class_chars"]):
        byte_to_class[ord(ch)] = idx

    X = byte_to_class[X_bytes].astype(np.int32)
    assert (X >= 0).all(), "unseen byte in training input"

    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    scale_E = math.sqrt(6.0 / (C + D))
    E = rng.uniform(-scale_E, scale_E, (d, C, D)).astype(np.float32)
    scale_W = math.sqrt(6.0 / (D + C))
    W = rng.uniform(-scale_W, scale_W, (D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def grad_ce(logits, y_true):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_true] -= 1.0
        return probs / B

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            h = np.sum(E[xb], axis=1)
            logits = h @ W + b
            g = grad_ce(logits, yb)

            grad_b = g.sum(axis=0)
            grad_W = h.T @ g
            grad_h = g @ W.T
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_h)

            grad_E += L2 * E
            grad_W += L2 * W

            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_E, grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x4C424D45
    header = struct.pack("<4i", magic, d, D, C)
    out = header + byte_to_class.astype(np.int8).tobytes() + E.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16 + 256) { fprintf(stderr, "bad weights: too small\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 64) { fprintf(stderr, "bad weights: dims\n"); return 3; }

    const int8_t *mapping = (const int8_t *)(wb + 16);
    const float *E = (const float *)(wb + 16 + 256);
    const float *W = E + (long)d * C * D;
    const float *b = W + (long)D * C;

    long expected = 16 + 256 + 4L * ((long)d * C * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(D, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            int8_t cls = mapping[xi[j]];
            if (cls < 0) { fprintf(stderr, "invalid byte\n"); return 3; }
            const float *e = E + ((long)j * C + cls) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < D; u++) sum += h[u] * W[(long)u * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```