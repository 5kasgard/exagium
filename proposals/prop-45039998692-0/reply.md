```hypothesis
Family: label-fm (Factorization Machine with shared embeddings and class-specific linear terms).
Why: Captures pairwise feature interactions via low-rank embeddings, crucial for tabular data like Adult, while keeping inference fast (O(d*D) per record). Shared embeddings keep model size small.
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear by >2×.
```

```python
"""label-fm: Factorization Machine for tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Encodes each of the 14 input bytes as a field with 256 values.
Learns shared embeddings (dim D) for FM interactions + class-specific linear weights.
Chooses D on last 20% of train.in (tuning split), then retrains on all train.in.
Exports: magic, d, D, C; embeddings[d][256][D]; linear[d][256][C]; bias[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on tuning split)
D_CANDIDATES = [4, 8, 16, 32]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """logits: (n, C), y: (n,) int. Returns dL/dlogits (n, C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return (probs / n).astype(np.float32)


def fm_forward(X, emb, lin, bias):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    C = bias.shape[0]
    D = emb.shape[2]
    logits = np.tile(bias, (n, 1))  # (n, C)
    # linear part
    for j in range(d):
        logits += lin[j, X[:, j], :]  # (n, C)
    # FM interaction (shared across classes)
    S = np.zeros((n, D), np.float32)
    Sq = np.zeros((n, D), np.float32)
    for j in range(d):
        e = emb[j, X[:, j], :]  # (n, D)
        S += e
        Sq += e * e
    inter = 0.5 * np.sum(S * S - Sq, axis=1, keepdims=True)  # (n, 1)
    logits += inter
    return logits


def fm_backward(X, emb, lin, bias, grad_logits):
    """Accumulate gradients into provided arrays (same shapes)."""
    n, d = X.shape
    C = bias.shape[0]
    D = emb.shape[2]
    # bias grad
    dbias = grad_logits.sum(axis=0)  # (C,)
    # linear grad
    dlin = np.zeros_like(lin)
    for j in range(d):
        for c in range(C):
            np.add.at(dlin[j, :, c], X[:, j], grad_logits[:, c])
    # embedding grad
    demb = np.zeros_like(emb)
    # compute S per sample
    S = np.zeros((n, D), np.float32)
    for j in range(d):
        S += emb[j, X[:, j], :]
    # sum of grad_logits over classes -> (n,)
    g_sum = grad_logits.sum(axis=1)  # (n,)
    for j in range(d):
        v = X[:, j]
        e = emb[j, v, :]  # (n, D)
        # grad = g_sum[:, None] * (S - e)
        demb[j, v, :] += (g_sum[:, None] * (S - e)).astype(np.float32)
    return dbias, dlin, demb


def train_model(X, y, D, epochs, rng):
    n, d = X.shape
    C = 2
    # init
    emb = rng.uniform(-1/math.sqrt(D), 1/math.sqrt(D), (d, 256, D)).astype(np.float32)
    lin = np.zeros((d, 256, C), np.float32)
    bias = np.zeros(C, np.float32)
    params = [emb, lin, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits = fm_forward(xb, emb, lin, bias)
            g = softmax_cross_entropy_grad(logits, yb)
            dbias, dlin, demb = fm_backward(xb, emb, lin, bias, g)
            # L2 regularization
            demb += L2 * emb
            dlin += L2 * lin
            dbias += L2 * bias
            grads = [demb, dlin, dbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, lin, bias


def evaluate(X, y, emb, lin, bias):
    logits = fm_forward(X, emb, lin, bias)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = D_CANDIDATES[0], -1.0
    tune_epochs = min(10, math.ceil(SAMPLES / len(X_train)))
    for D in D_CANDIDATES:
        emb, lin, bias = train_model(X_train, y_train, D, tune_epochs, rng)
        acc = evaluate(X_tune, y_tune, emb, lin, bias)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # final training on all data
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    emb, lin, bias = train_model(X_all, y_all, best_D, final_epochs, rng)

    # pack weights
    d, C = X_all.shape[1], card["classes"]
    magic = 0x464D3131  # 'FM11' LE
    header = struct.pack("<4i", magic, d, best_D, C)
    # embeddings: d * 256 * D floats
    emb_bytes = emb.astype("<f4").tobytes()
    # linear: d * 256 * C floats
    lin_bytes = lin.astype("<f4").tobytes()
    # bias: C floats
    bias_bytes = bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + lin_bytes + bias_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per field). Output: 1 byte class index. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + (long)d * 256 * D;
    const float *bias = lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * D);
    float *Sq = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(S, 0, D * sizeof(float));
        memset(Sq, 0, D * sizeof(float));
        for (int c = 0; c < C; c++) score[c] = bias[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((long)j * 256 + v) * D;
            const float *l = lin + ((long)j * 256 + v) * C;
            for (int kk = 0; kk < D; kk++) {
                float val = e[kk];
                S[kk] += val;
                Sq[kk] += val * val;
            }
            for (int c = 0; c < C; c++) score[c] += l[c];
        }
        float inter = 0.0f;
        for (int kk = 0; kk < D; kk++) inter += S[kk]*S[kk] - Sq[kk];
        inter *= 0.5f;
        for (int c = 0; c < C; c++) score[c] += inter;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```