```hypothesis
The label-embed family factorizes the linear model's weight matrix into per-position byte embeddings (d × 256 × D) and class embeddings (C × D). This captures pairwise position–byte interactions at inference cost O(d·D + C·D), far below label-linear's O(d·256·C). On TinyShakespeare, character dependencies span multiple positions, so a low-rank interaction model should beat the purely additive linear baseline. If dev accuracy does not exceed label-linear (0.3375), the rank D is too small or the task gains little from interactions.
```

```python
"""label-embed: per-position byte embeddings + class embeddings (low-rank factorization).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
BATCH = 64
EPOCHS_TUNE = 8
EPOCHS_FULL = 20
D_CANDIDATES = [8, 16, 24, 32, 48, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(scores, y):
    return (scores.argmax(axis=1) == y).mean()


def train_eval(D, X_tr, y_tr, X_te, y_te, epochs, rng):
    n, d = X_tr.shape
    C = 65
    # params
    pos_emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    cls_emb = rng.normal(0, 0.01, (C, D)).astype(np.float32)
    cls_bias = np.zeros(C, np.float32)
    # Adam state
    m_pos, v_pos = np.zeros_like(pos_emb), np.zeros_like(pos_emb)
    m_cls, v_cls = np.zeros_like(cls_emb), np.zeros_like(cls_emb)
    m_bias, v_bias = np.zeros_like(cls_bias), np.zeros_like(cls_bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bsz = len(yb)
            # forward: context embedding = sum_j pos_emb[j, xb[:,j]]
            ctx = np.zeros((bsz, D), np.float32)
            for j in range(d):
                ctx += pos_emb[j, xb[:, j]]
            logits = ctx @ cls_emb.T + cls_bias  # (bsz, C)
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz
            # gradients
            g_cls_emb = ctx.T @ probs  # (D, C)
            g_cls_bias = probs.sum(axis=0)
            g_ctx = probs @ cls_emb  # (bsz, D)
            g_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                # scatter-add g_ctx into g_pos_emb[j, xb[:,j]]
                np.add.at(g_pos_emb[j], xb[:, j], g_ctx)
            # Adam update
            t += 1
            for param, grad, m, v in [
                (pos_emb, g_pos_emb, m_pos, v_pos),
                (cls_emb, g_cls_emb.T, m_cls, v_cls),
                (cls_bias, g_cls_bias, m_bias, v_bias),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
    # evaluate
    ctx_te = np.zeros((len(X_te), D), np.float32)
    for j in range(d):
        ctx_te += pos_emb[j, X_te[:, j]]
    logits_te = ctx_te @ cls_emb.T + cls_bias
    return accuracy(logits_te, y_te), (pos_emb, cls_emb, cls_bias)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = D_CANDIDATES[0], -1.0
    best_params = None
    for D in D_CANDIDATES:
        rng_d = np.random.default_rng(SEED + D)  # deterministic per D
        acc, params = train_eval(D, X_tr, y_tr, X_te, y_te, EPOCHS_TUNE, rng_d)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params

    # final training on all data
    rng_full = np.random.default_rng(SEED + 1000 + best_D)
    _, final_params = train_eval(best_D, X, y, X[:1], y[:1], EPOCHS_FULL, rng_full)
    pos_emb, cls_emb, cls_bias = final_params

    # write weights.bin: magic, d, D, C, pos_emb[d][256][D], cls_emb[C][D], cls_bias[C]
    magic = 0x44424D45  # "EMBD"
    out = struct.pack("<4i", magic, d, best_D, C)
    out += pos_emb.astype("<f4").tobytes()
    out += cls_emb.astype("<f4").tobytes()
    out += cls_bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index. */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)C * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *pos_emb = (const float *)(wb + 16);
    const float *cls_emb = pos_emb + (long)d * 256 * D;
    const float *cls_bias = cls_emb + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *ctx = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(ctx, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *pe = pos_emb + ((long)j * 256 + xi[j]) * D;
            for (int k = 0; k < D; k++) ctx[k] += pe[k];
        }
        for (int c = 0; c < C; c++) {
            float sum = cls_bias[c];
            const float *ce = cls_emb + (long)c * D;
            for (int k = 0; k < D; k++) sum += ctx[k] * ce[k];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```