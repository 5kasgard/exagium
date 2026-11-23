```hypothesis
Family: label-posemb (positional embedding sum + linear). Each of the 16 context positions gets a 256×D embedding table; the 16 vectors are summed and fed to a linear layer (D→65). This captures positional n-gram statistics with far fewer parameters than one-hot linear (16·256·D + D·65 vs 4096·65) and should generalize better on character-level language modeling. If dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 7359 Ir/item, the hypothesis fails.
```

```python
"""Positional embedding sum + linear classifier for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x50454D42, int32 d=16, D, C=65; float32 E[16][256][D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid searched on the last 20% of train.in (tuning split)
D_CHOICES = [8, 16, 32, 64]
LR_CHOICES = [0.03, 0.01, 0.003]
EPOCH_CHOICES = [5, 10, 20]
BATCH = 64
L2 = 1e-5
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_eval(X_tr, y_tr, X_va, y_va, D, lr, epochs, rng):
    n_tr, d = X_tr.shape
    C = 65
    # params
    E = (rng.normal(0, 1/math.sqrt(D), (d, 256, D)).astype(np.float32))
    W = np.zeros((D, C), np.float32)
    b = np.zeros(C, np.float32)
    # adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            h = np.zeros((len(yb), D), np.float32)
            for j in range(d):
                h += E[j, xb[:, j]]
            logits = h @ W + b
            # softmax cross-entropy gradient
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # gradients
            gW = h.T @ probs + L2 * W
            gb = probs.sum(axis=0)
            gh = probs @ W.T
            gE = np.zeros_like(E)
            for j in range(d):
                np.add.at(gE[j], xb[:, j], gh)
            gE += L2 * E
            # adam update
            t += 1
            for param, grad, m, v in [(E, gE, mE, vE), (W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= ADAM_BETA1; m += (1 - ADAM_BETA1) * grad
                v *= ADAM_BETA2; v += (1 - ADAM_BETA2) * grad * grad
                m_hat = m / (1 - ADAM_BETA1 ** t)
                v_hat = v / (1 - ADAM_BETA2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
    # validation accuracy
    h_va = np.zeros((len(y_va), D), np.float32)
    for j in range(d):
        h_va += E[j, X_va[:, j]]
    logits_va = h_va @ W + b
    return accuracy(logits_va, y_va), (E, W, b)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65

    # tuning split: last 20% in file order
    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_hp = -1, None, None
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCH_CHOICES:
                # deterministic per trial
                trial_rng = np.random.default_rng(SEED + hash((D, lr, epochs)) % 2**32)
                acc, params = train_eval(X_tr, y_tr, X_va, y_va, D, lr, epochs, trial_rng)
                if acc > best_acc:
                    best_acc, best_params, best_hp = acc, params, (D, lr, epochs)
    print(f"Best hp: D={best_hp[0]}, lr={best_hp[1]}, epochs={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best hp
    D, lr, epochs = best_hp
    full_rng = np.random.default_rng(SEED)
    _, best_params = train_eval(X, y, X[:1], y[:1], D, lr, epochs, full_rng)  # dummy val
    E, W, b = best_params

    # write weights.bin
    magic = 0x50454D42  # "PEMB"
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Positional embedding sum + linear classifier: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x50454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + (size_t)d * 256 * D * 4 + (size_t)D * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (size_t)d * 256 * D;
    const float *b = W + (size_t)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(D, sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld: %u bytes, expected %d\n", i, k, d); return 3; }
        memset(h, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + ((size_t)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < D; u++) sum += h[u] * W[(size_t)u * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```