```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset exhibits strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that a linear model cannot capture. An FM with embedding dimension k=8 models all 91 feature pairs using only ~32K parameters (d=14, 256 values each) and evaluates in O(d·k) time via the summed-embeddings trick, keeping inference cost near the linear baseline. Failure if dev accuracy ≤ 0.8524 (label-linear) or instruction cost ≥ 2× label-linear (~1400 Ir/item).
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
L2_EMB = 1e-4
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

def train_fm(X, y, d, k, epochs, rng, bias_init=None, linear_init=None, embed_init=None):
    n = X.shape[0]
    if bias_init is None:
        bias = np.float32(0.0)
        linear = np.zeros((d, 256), dtype=np.float32)
        embed = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    else:
        bias, linear, embed = bias_init, linear_init, embed_init

    m_bias, v_bias = 0.0, 0.0
    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_embed = np.zeros_like(embed)
    v_embed = np.zeros_like(embed)

    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            B = len(yi)

            scores = np.full(B, bias, dtype=np.float32)
            sum_emb = np.zeros((B, k), dtype=np.float32)
            sum_emb_sq = np.zeros((B, k), dtype=np.float32)

            for j in range(d):
                v = xi[:, j]
                scores += linear[j, v]
                e = embed[j, v, :]
                sum_emb += e
                sum_emb_sq += e * e

            interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            scores += interaction

            sig = 1.0 / (1.0 + np.exp(-scores))
            grad_score = (sig - yi).astype(np.float32)

            dbias = grad_score.mean()

            dlinear = np.zeros_like(linear)
            dembed = np.zeros_like(embed)

            for j in range(d):
                v = xi[:, j]
                np.add.at(dlinear[j], v, grad_score)
                sum_grad = np.zeros(256, dtype=np.float32)
                np.add.at(sum_grad, v, grad_score)
                for f in range(k):
                    weighted = grad_score * sum_emb[:, f]
                    np.add.at(dembed[j, :, f], v, weighted)
                dembed[j] -= embed[j] * sum_grad[:, None]

            dlinear = (dlinear / B) + L2 * linear
            dembed = (dembed / B) + L2_EMB * embed

            t += 1
            m_bias = 0.9 * m_bias + 0.1 * dbias
            v_bias = 0.999 * v_bias + 0.001 * dbias * dbias
            m_hat = m_bias / (1 - 0.9 ** t)
            v_hat = v_bias / (1 - 0.999 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            m_linear = 0.9 * m_linear + 0.1 * dlinear
            v_linear = 0.999 * v_linear + 0.001 * dlinear * dlinear
            m_hat = m_linear / (1 - 0.9 ** t)
            v_hat = v_linear / (1 - 0.999 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            m_embed = 0.9 * m_embed + 0.1 * dembed
            v_embed = 0.999 * v_embed + 0.001 * dembed * dembed
            m_hat = m_embed / (1 - 0.9 ** t)
            v_hat = v_embed / (1 - 0.999 ** t)
            embed -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return bias, linear, embed

def eval_fm(X, y, d, k, bias, linear, embed):
    n = X.shape[0]
    scores = np.full(n, bias, dtype=np.float32)
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_emb_sq = np.zeros((n, k), dtype=np.float32)

    for j in range(d):
        v = X[:, j]
        scores += linear[j, v]
        e = embed[j, v, :]
        sum_emb += e
        sum_emb_sq += e * e

    interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    scores += interaction
    preds = (scores > 0).astype(np.int64)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_k = 8
    best_acc = 0.0
    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / X_train.shape[0]))

    for k in (4, 8, 16):
        rng_k = np.random.default_rng(SEED + k)
        bias, linear, embed = train_fm(X_train, y_train, d, k, epochs_tune, rng_k)
        acc = eval_fm(X_val, y_val, d, k, bias, linear, embed)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_full = np.random.default_rng(SEED)
    bias, linear, embed = train_fm(X, y, d, best_k, epochs_full, rng_full)

    out = struct.pack("<4i", 0x314D46, d, best_k, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x314D46) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = bias;
        for (int f = 0; f < k; f++) {
            sum_emb[f] = 0.0f;
            sum_emb_sq[f] = 0.0f;
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = embed + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }

        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```