```hypothesis
Family: label-emb. Each categorical feature gets a learned embedding (size E=4); each ordinal feature is z-scored and linearly projected to E=4. The 14 E-d vectors are concatenated and passed to a linear output layer. This adds feature-specific non-linearity (via embeddings) while keeping inference extremely cheap (lookups + one small matmul). Expected to beat label-linear (0.8524) on dev accuracy at lower cost than label-mlp16. Failure if dev accuracy ≤0.8524 or Ir/item > 2000.
```
```python
"""label-emb: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 4                 # embedding/projection size per feature
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4231    # 'EMB1'

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

    # --- per-feature metadata ---
    ord_mean = np.zeros(d, np.float32)
    ord_std  = np.ones(d, np.float32)
    cat_sizes = np.zeros(d, np.int32)
    cat_maps = [None]*d          # for categorical: map byte value -> index (unseen = last)
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_mean[j], ord_std[j] = mu, sd
        else:
            vals = np.unique(X[:, j])
            cat_sizes[j] = len(vals) + 1   # +1 for unseen (255)
            mp = np.full(256, cat_sizes[j]-1, dtype=np.int32)
            mp[vals] = np.arange(len(vals))
            cat_maps[j] = mp

    # --- parameter allocation ---
    # ordinal: W_ord[d_ord, E], b_ord[d_ord, E]
    # categorical: Emb[j][cat_sizes[j], E]
    # output: W_out[d*E, C], b_out[C]
    d_ord = sum(1 for t in types if t == "ordinal")
    W_ord = rng.normal(0, 0.01, (d_ord, E)).astype(np.float32)
    b_ord = np.zeros((d_ord, E), np.float32)
    Emb = []
    for j in range(d):
        if types[j] == "categorical":
            Emb.append(rng.normal(0, 0.01, (cat_sizes[j], E)).astype(np.float32))
        else:
            Emb.append(None)
    W_out = rng.normal(0, math.sqrt(6.0/(d*E + C)), (d*E, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [W_ord, b_ord, W_out, b_out] + [e for e in Emb if e is not None]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # --- training loop ---
    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            f = np.zeros((bs, d*E), np.float32)
            oi = 0
            for j in range(d):
                if types[j] == "ordinal":
                    z = (xb[:, j].astype(np.float32) - ord_mean[j]) / ord_std[j]   # (bs,)
                    f[:, j*E:(j+1)*E] = z[:, None] * W_ord[oi] + b_ord[oi]
                    oi += 1
                else:
                    mp = cat_maps[j]
                    idxs = mp[xb[:, j]]
                    f[:, j*E:(j+1)*E] = Emb[j][idxs]

            logits = f @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # loss gradient
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs

            # backward
            grad_f = grad_logits @ W_out.T
            grad_W_out = f.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(0)

            grad_W_ord = np.zeros_like(W_ord)
            grad_b_ord = np.zeros_like(b_ord)
            grad_Emb = [np.zeros_like(e) if e is not None else None for e in Emb]

            oi = 0
            for j in range(d):
                gj = grad_f[:, j*E:(j+1)*E]
                if types[j] == "ordinal":
                    z = (xb[:, j].astype(np.float32) - ord_mean[j]) / ord_std[j]
                    grad_W_ord[oi] = (z[:, None] * gj).sum(0) + L2 * W_ord[oi]
                    grad_b_ord[oi] = gj.sum(0)
                    oi += 1
                else:
                    idxs = cat_maps[j][xb[:, j]]
                    np.add.at(grad_Emb[j], idxs, gj)

            grads = [grad_W_ord, grad_b_ord, grad_W_out, grad_b_out] + [g for g in grad_Emb if g is not None]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- pack weights.bin ---
    # header: magic, d, E, C
    # per feature:
    #   if ordinal: type=0, mean, std, W_ord[E], b_ord[E]
    #   if categorical: type=1, cat_size, Emb[cat_size*E]
    # output: W_out[d*E*C], b_out[C]
    out = bytearray()
    out += struct.pack("<4i", MAGIC, d, E, C)
    oi = 0
    for j in range(d):
        if types[j] == "ordinal":
            out += struct.pack("<B", 0)
            out += struct.pack("<ff", ord_mean[j], ord_std[j])
            out += W_ord[oi].astype("<f4").tobytes()
            out += b_ord[oi].astype("<f4").tobytes()
            oi += 1
        else:
            out += struct.pack("<Bi", 1, cat_sizes[j])
            out += Emb[j].astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x454D4231) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int E = (int)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    int C = (int)(wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24);
    if (d <= 0 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(d * E * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    const unsigned char *p = wb + 16;

    /* skip training metadata, find output layer */
    for (int j = 0; j < d; j++) {
        uint8_t type = *p++; p++;
        if (type == 0) { /* ordinal: mean, std, W_ord[E], b_ord[E] */
            p += 2*sizeof(float) + 2*E*sizeof(float);
        } else { /* categorical: cat_size, Emb[cat_size*E] */
            uint32_t cat_size = *((uint32_t*)p); p += 4;
            p += cat_size * E * sizeof(float);
        }
    }
    const float *W_out = (const float*)p; p += (long)d * E * C * sizeof(float);
    const float *b_out = (const float*)p;

    /* rewind to read per-feature params */
    p = wb + 16;
    float *ord_mean = malloc(d * sizeof(float));
    float *ord_std  = malloc(d * sizeof(float));
    float *W_ord = malloc(d * E * sizeof(float));  /* only ordinal entries used */
    float *b_ord = malloc(d * E * sizeof(float));
    const float **Emb = malloc(d * sizeof(float*));
    int *cat_size = malloc(d * sizeof(int));
    int ord_idx = 0;

    for (int j = 0; j < d; j++) {
        uint8_t type = *p++; p++;
        if (type == 0) {
            ord_mean[j] = *((float*)p); p += sizeof(float);
            ord_std[j]  = *((float*)p); p += sizeof(float);
            memcpy(&W_ord[ord_idx * E], p, E * sizeof(float)); p += E * sizeof(float);
            memcpy(&b_ord[ord_idx * E], p, E * sizeof(float)); p += E * sizeof(float);
            Emb[j] = NULL;
            cat_size[j] = 0;
            ord_idx++;
        } else {
            cat_size[j] = *((uint32_t*)p); p += 4;
            Emb[j] = (const float*)p; p += (long)cat_size[j] * E * sizeof(float);
            ord_mean[j] = ord_std[j] = 0;
        }
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        ord_idx = 0;
        for (int j = 0; j < d; j++) {
            float *fj = feat + j * E;
            if (Emb[j] == NULL) { /* ordinal */
                float z = (xi[j] - ord_mean[j]) / ord_std[j];
                for (int e = 0; e < E; e++) fj[e] = z * W_ord[ord_idx * E + e] + b_ord[ord_idx * E + e];
                ord_idx++;
            } else { /* categorical */
                int idx = xi[j];
                if (idx >= cat_size[j]) idx = cat_size[j] - 1; /* unseen */
                const float *emb = Emb[j] + (long)idx * E;
                for (int e = 0; e < E; e++) fj[e] = emb[e];
            }
        }

        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int fidx = 0; fidx < d * E; fidx++) {
            float fv = feat[fidx];
            if (fv != 0.0f) {
                for (int c = 0; c < C; c++) logits[c] += fv * W_out[(long)fidx * C + c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```