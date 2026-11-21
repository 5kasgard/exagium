```hypothesis
Family: label-ensemble (bag of 3 linear models with different seeds). 
Why: Character prediction benefits from variance reduction; an ensemble of independently initialized linear models (same architecture as label-linear) should boost accuracy significantly while keeping inference simple (sum logits, argmax). Cost ~3× label-linear, still far below GBDT.
Failure signal: dev accuracy ≤ 0.34 or instructions/item > 30,000.
```

```python
import json, math, struct, sys
import numpy as np

M = 3
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

    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    all_models = []
    for m in range(M):
        rng = np.random.default_rng(SEED + m)
        W1 = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t_step = 0
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                z, yy = Z[idx], y[idx]
                S = z @ W1 + b1
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
                t_step += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    mv, vv = state[i]
                    mv *= 0.9; mv += 0.1 * gr
                    vv *= 0.999; vv += 0.001 * gr * gr
                    p -= LR * (mv / (1 - 0.9 ** t_step)) / (np.sqrt(vv / (1 - 0.999 ** t_step)) + 1e-8)
        off = 0
        T_list = []
        for E in enc:
            T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
            off += E.shape[1]
        all_models.append((T_list, b1.astype(np.float32)))

    magic = 0x454E534D
    H = 0
    header = struct.pack("<5i", magic, d, H, C, M)
    out = header
    for T_list, b1 in all_models:
        T_arr = np.stack(T_list, axis=0)
        out += T_arr.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454E534D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], M = hd[4];
    int W = H ? H : C;
    if (C < 1 || C > 256 || d != 16 || H != 0 || M != 3) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20 + (long)M * ((long)d * 256 * W + W) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *base = (const float *)(hd + 5);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *acc = malloc(sizeof(float) * W);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = 0.0f;

        const float *model_ptr = base;
        for (int m = 0; m < M; m++) {
            const float *T = model_ptr;
            const float *b1 = T + (long)d * 256 * W;
            model_ptr = b1 + W;

            for (int w = 0; w < W; w++) acc[w] = b1[w];
            for (int j = 0; j < d; j++) {
                const float *t = T + ((long)j * 256 + xi[j]) * W;
                for (int w = 0; w < W; w++) acc[w] += t[w];
            }
            for (int c = 0; c < C; c++) scores[c] += acc[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```