```hypothesis
Family: label-pemb (Positional Embedding Model). Each of the 16 context positions gets a learned byte embedding (dimension D=2..8), concatenated and fed to a linear classifier. This captures byte similarities (e.g., letters share patterns) with far fewer parameters than one-hot linear (266k → ~20k), improving generalization on character prediction. If dev accuracy ≤ 0.3375 (label-linear), the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
GRID_D = [2, 4, 8]
GRID_LR = [0.03, 0.01, 0.003]
BATCH = 64
L2 = 1e-4
TUNE_EPOCHS = 10
FULL_EPOCHS = 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_loop(X, y, d, D, C, lr, epochs, rng, params=None):
    if params is None:
        emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
        W = rng.normal(0, 0.01, (C, d * D)).astype(np.float32)
        b = np.zeros(C, np.float32)
    else:
        emb, W, b = params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in (emb, W, b)]
    t = 0
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            emb_vals = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)  # (B, d, D)
            flat = emb_vals.reshape(len(yb), -1)  # (B, d*D)
            logits = flat @ W.T + b
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)  # (B, C)
            # grads
            grad_W = grad_logits.T @ flat + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ W  # (B, d*D)
            grad_emb_vals = grad_flat.reshape(len(yb), d, D)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_vals[:, j, :])
            grad_emb += L2 * emb
            # Adam
            t += 1
            for i, (p, gr) in enumerate(zip((emb, W, b), (grad_emb, grad_W, grad_b))):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W, b


def eval_acc(X, y, emb, W, b):
    d, D = emb.shape[0], emb.shape[2]
    emb_vals = emb[np.arange(d)[:, None], X.T].transpose(1, 0, 2)
    flat = emb_vals.reshape(len(y), -1)
    logits = flat @ W.T + b
    return (logits.argmax(axis=1) == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = len(X)
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_acc = -1.0
    best_cfg = None
    for D in GRID_D:
        for lr in GRID_LR:
            cfg_seed = (D * 1000 + int(lr * 10000)) & 0xFFFFFFFF
            rng = np.random.default_rng(cfg_seed)
            emb, W, b = train_loop(X_tr, y_tr, d, D, C, lr, TUNE_EPOCHS, rng)
            acc = eval_acc(X_va, y_va, emb, W, b)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, lr)
    print(f"Best config: D={best_cfg[0]}, lr={best_cfg[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    D, lr = best_cfg
    rng = np.random.default_rng(SEED)
    emb, W, b = train_loop(X, y, d, D, C, lr, FULL_EPOCHS, rng)

    out = struct.pack("<4i", 0x50454D31, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x50454D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)C * d * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * d * D;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * d * D);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, sizeof(float) * d * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            float *a = acc + j * D;
            for (int kk = 0; kk < D; kk++) a[kk] += e[kk];
        }
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int m = 0; m < d * D; m++) {
            float am = acc[m];
            if (am != 0.0f) {
                const float *w = W + (long)m;
                for (int c = 0; c < C; c++) sc[c] += am * w[c * d * D];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```