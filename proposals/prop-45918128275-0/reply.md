```hypothesis
Family: label-emb-linear. Small per-category embeddings (E=4) for categorical features and linear projections for ordinal features are summed into a 4-d vector, then classified by a linear layer. This adds non-linear capacity via embeddings while keeping inference extremely cheap (only a few adds/multiplies per feature). It should beat label-linear (0.8524) on accuracy with similar or lower CPU cost. Failure: dev accuracy ≤ label-linear or instructions/item > label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
E = 4
LR = 0.01
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    cat_embs = []
    ord_ws = []
    cat_indices = []
    ord_indices = []
    for j, t in enumerate(types):
        if t == "categorical":
            unseen_emb = rng.normal(0, 0.01, size=E).astype(np.float32)
            emb_table = np.tile(unseen_emb, (256, 1))
            vals = np.unique(X[:, j])
            for v in vals:
                emb_table[v] = rng.normal(0, 0.01, size=E).astype(np.float32)
            cat_embs.append(emb_table)
            cat_indices.append(j)
        else:
            ord_w = rng.normal(0, 0.01, size=E).astype(np.float32)
            ord_ws.append(ord_w)
            ord_indices.append(j)

    V = rng.normal(0, 0.01, size=(E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = cat_embs + ord_ws + [V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            acc = np.zeros((bs, E), dtype=np.float32)
            for ci, j in enumerate(cat_indices):
                acc += cat_embs[ci][xb[:, j]]
            for oi, j in enumerate(ord_indices):
                x_val = xb[:, j].astype(np.float32)[:, None]
                acc += x_val * ord_ws[oi]

            logits = acc @ V + b
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs

            grad_V = acc.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)
            grad_acc = grad_logits @ V.T

            grad_cat_embs = [np.zeros_like(e) for e in cat_embs]
            grad_ord_ws = [np.zeros_like(w) for w in ord_ws]

            for ci, j in enumerate(cat_indices):
                np.add.at(grad_cat_embs[ci], xb[:, j], grad_acc)

            for oi, j in enumerate(ord_indices):
                x_val = xb[:, j].astype(np.float32)[:, None]
                grad_ord_ws[oi] = (grad_acc * x_val).sum(axis=0)

            for g, p in zip(grad_cat_embs, cat_embs): g += L2 * p
            for g, p in zip(grad_ord_ws, ord_ws): g += L2 * p
            grad_V += L2 * V

            grads = grad_cat_embs + grad_ord_ws + [grad_V, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x454D424C
    out = struct.pack("<4I", magic, d, E, C)
    for j in range(d):
        if types[j] == "categorical":
            ci = cat_indices.index(j)
            out += struct.pack("<B", 1)
            out += cat_embs[ci].astype("<f4").tobytes()
        else:
            oi = ord_indices.index(j)
            out += struct.pack("<B", 0)
            out += ord_ws[oi].astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    uint32_t magic = hdr[0];
    if (magic != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1];
    int E = hdr[2];
    int C = hdr[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }

    const unsigned char *ptr = wb + 16;
    const unsigned char *end = wb + nw;

    int *feat_type = malloc(d * sizeof(int));
    const float **feat_data = malloc(d * sizeof(float*));

    for (int j = 0; j < d; j++) {
        if (ptr + 1 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint8_t type = *ptr++;
        feat_type[j] = type;
        feat_data[j] = (const float *)ptr;
        if (type == 1) {
            size_t bytes = 256 * (size_t)E * sizeof(float);
            if (ptr + bytes > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            ptr += bytes;
        } else {
            size_t bytes = (size_t)E * sizeof(float);
            if (ptr + bytes > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            ptr += bytes;
        }
    }
    const float *V = (const float *)ptr;
    ptr += (size_t)E * C * sizeof(float);
    const float *b = (const float *)ptr;
    ptr += (size_t)C * sizeof(float);
    if (ptr != end) { fprintf(stderr, "extra data in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            if (feat_type[j] == 1) {
                uint8_t cat = xi[j];
                const float *emb = feat_data[j] + (size_t)cat * E;
                for (int e = 0; e < E; e++) acc[e] += emb[e];
            } else {
                float val = (float)xi[j];
                const float *w = feat_data[j];
                for (int e = 0; e < E; e++) acc[e] += val * w[e];
            }
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int e = 0; e < E; e++) sum += acc[e] * V[(size_t)e * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```