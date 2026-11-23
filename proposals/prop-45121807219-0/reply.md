```hypothesis
Factorization Machines (FM) capture pairwise feature interactions with a small latent dimension (k=4), which is crucial for tabular data like Adult where interactions (e.g., education×occupation) matter. With k=4, inference is O(d·k) ≈ 56 multiply‑adds, faster than mlp16 and competitive with linear. If dev accuracy does not exceed linear, it shows interactions are not helpful or k is too small.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, d, k, lr, l2, epochs, batch, rng, X_val=None, y_val=None):
    n = X.shape[0]
    w0 = 0.0
    lw = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_lw, v_lw = np.zeros_like(lw), np.zeros_like(lw)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            yb = y[idx]
            bsz = len(idx)

            linear = w0 + lw[np.arange(d)[None, :], xb].sum(axis=1)
            emb_batch = emb[np.arange(d)[None, :], xb]          # (bsz, d, k)
            sum_emb = emb_batch.sum(axis=1)                     # (bsz, k)
            sum_emb_sq = (emb_batch ** 2).sum(axis=1)           # (bsz, k)
            interaction = 0.5 * ((sum_emb ** 2).sum(axis=1) - sum_emb_sq.sum(axis=1))
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            g = (p - yb).astype(np.float32)                     # (bsz,)

            gw0 = g.sum()
            gl = g                                               # (bsz,)
            gi = g                                               # (bsz,)

            glw = np.zeros((d, 256), dtype=np.float32)
            gemb = np.zeros((d, 256, k), dtype=np.float32)

            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        glw[j, v] = gl[mask].sum()
                for i in range(bsz):
                    v = vals[i]
                    ev = emb_batch[i, j]
                    diff = sum_emb[i] - ev
                    gemb[j, v] += gi[i] * diff

            glw += l2 * lw
            gemb += l2 * emb

            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * gw0
            v_w0 = 0.999 * v_w0 + 0.001 * gw0 * gw0
            w0 -= lr * (m_w0 / (1 - 0.9 ** t)) / (np.sqrt(v_w0 / (1 - 0.999 ** t)) + 1e-8)
            # lw
            m_lw = 0.9 * m_lw + 0.1 * glw
            v_lw = 0.999 * v_lw + 0.001 * glw * glw
            lw -= lr * (m_lw / (1 - 0.9 ** t)) / (np.sqrt(v_lw / (1 - 0.999 ** t)) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * gemb
            v_emb = 0.999 * v_emb + 0.001 * gemb * gemb
            emb -= lr * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    if X_val is not None:
        linear_val = w0 + lw[np.arange(d)[None, :], X_val].sum(axis=1)
        emb_val = emb[np.arange(d)[None, :], X_val]
        sum_emb_val = emb_val.sum(axis=1)
        sum_emb_sq_val = (emb_val ** 2).sum(axis=1)
        interaction_val = 0.5 * ((sum_emb_val ** 2).sum(axis=1) - sum_emb_sq_val.sum(axis=1))
        logit_val = linear_val + interaction_val
        pred_val = (logit_val > 0).astype(np.int64)
        val_acc = (pred_val == y_val).mean()
        return w0, lw, emb, val_acc
    return w0, lw, emb, None

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    assert card["classes"] == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for k in (2, 4, 8):
        for lr in (0.01, 0.03):
            for l2 in (1e-4, 1e-3):
                epochs = min(100, math.ceil(600000 / len(X_tr)))
                w0, lw, emb, acc = train_fm(X_tr, y_tr, d, k, lr, l2, epochs, 64, rng, X_val, y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, l2)
                    best_model = (w0, lw, emb)

    k, lr, l2 = best_params
    epochs = min(100, math.ceil(600000 / n))
    w0, lw, emb, _ = train_fm(X, y, d, k, lr, l2, epochs, 64, rng)

    magic = 0x464D3130  # "FM10"
    out = struct.pack("<iii", magic, d, k)
    out += struct.pack("<f", w0)
    out += lw.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    if (nw < 12 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 12 + 4 + 4L * d * 256 * (1 + k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 3);
    float w0 = p[0];
    const float *lw = p + 1;
    const float *emb = lw + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(k, sizeof(float));
    float *sum_emb_sq = calloc(k, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float linear = w0;
        memset(sum_emb, 0, k * sizeof(float));
        memset(sum_emb_sq, 0, k * sizeof(float));

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            long idx = (long)j * 256 + v;
            linear += lw[idx];
            const float *e = emb + idx * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
        inter *= 0.5f;
        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```