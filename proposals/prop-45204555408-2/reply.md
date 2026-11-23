```hypothesis
Family: label-embsum (positional embedding sum + linear). Each of the 16 context positions has a 256×D embedding table; the 16 embeddings are summed to a D-vector, then a linear layer predicts 65 classes. This captures positional n-gram statistics with far fewer parameters than one-hot linear (16×256×D + D×65 vs 16×256×65) and should match or beat label-linear accuracy at lower inference cost. If dev accuracy ≤ 0.335 or inference > 1000 Ir/item, the hypothesis fails.
```
```python
"""label-embsum: positional embedding sum + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Architecture: for each position j in 0..15, embedding table E_j[256][D]. Sum_j E_j[x_j] -> h (D).
Logits = h @ W + b, W[D][C], b[C]. Cross-entropy, Adam, L2.
Weights format: magic 0x454D4253, int32 d=16, D, C=65; float32 E[16][256][D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
D_CHOICES = [8, 16, 32]
LR_CHOICES = [0.03, 0.1, 0.01]
L2_CHOICES = [0.0, 1e-4]
EPOCHS_GRID = 10
EPOCHS_FULL = 20
BATCH = 64
SAMPLES = 600000  # max samples per grid run


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_step(Xb, yb, E, W, b, lr, l2, opt_state, step):
    """Adam update for one minibatch. Returns loss."""
    n = Xb.shape[0]
    # forward
    h = np.zeros((n, E.shape[2]), dtype=np.float32)
    for j in range(16):
        h += E[j, Xb[:, j]]
    logits = h @ W + b
    # softmax cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), yb]).mean()
    # gradients
    grad_logits = probs.copy()
    grad_logits[np.arange(n), yb] -= 1.0
    grad_logits /= n
    grad_W = h.T @ grad_logits + l2 * W
    grad_b = grad_logits.sum(axis=0)
    grad_h = grad_logits @ W.T
    grad_E = np.zeros_like(E)
    for j in range(16):
        np.add.at(grad_E[j], Xb[:, j], grad_h)
    grad_E += l2 * E
    # Adam update
    t = step + 1
    for param, grad, (m, v) in zip([E, W, b], [grad_E, grad_W, grad_b], opt_state):
        m *= 0.9; m += 0.1 * grad
        v *= 0.999; v += 0.001 * grad * grad
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return loss


def evaluate(X, y, E, W, b):
    n = X.shape[0]
    h = np.zeros((n, E.shape[2]), dtype=np.float32)
    for j in range(16):
        h += E[j, X[:, j]]
    logits = h @ W + b
    return accuracy(logits, y)


def run_grid(X_tr, y_tr, X_va, y_va, rng):
    best_acc = -1.0
    best_cfg = None
    best_params = None
    n_tr = X_tr.shape[0]
    steps_per_epoch = math.ceil(n_tr / BATCH)
    max_steps = min(EPOCHS_GRID * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # init
                E = (rng.uniform(-1, 1, (16, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
                W = (rng.uniform(-1, 1, (D, 65)) * math.sqrt(6.0 / D)).astype(np.float32)
                b = np.zeros(65, dtype=np.float32)
                opt_state = [(np.zeros_like(E), np.zeros_like(E)),
                             (np.zeros_like(W), np.zeros_like(W)),
                             (np.zeros_like(b), np.zeros_like(b))]
                step = 0
                order = np.arange(n_tr)
                for _ in range(EPOCHS_GRID):
                    if step >= max_steps:
                        break
                    rng.shuffle(order)
                    for s in range(0, n_tr, BATCH):
                        if step >= max_steps:
                            break
                        idx = order[s:s + BATCH]
                        train_step(X_tr[idx], y_tr[idx], E, W, b, lr, l2, opt_state, step)
                        step += 1
                acc = evaluate(X_va, y_va, E, W, b)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (D, lr, l2)
                    best_params = (E.copy(), W.copy(), b.copy())
    return best_cfg, best_params, best_acc


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 16 and card["classes"] == 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    tr_size = n - tune_size
    X_tr, y_tr = X[:tr_size], y[:tr_size]
    X_va, y_va = X[tr_size:], y[tr_size:]

    rng = np.random.default_rng(SEED)
    # grid search on tuning split
    (D, lr, l2), (E, W, b), _ = run_grid(X_tr, y_tr, X_va, y_va, rng)
    # retrain on full data
    n_full = n
    steps_per_epoch = math.ceil(n_full / BATCH)
    max_steps = min(EPOCHS_FULL * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    opt_state = [(np.zeros_like(E), np.zeros_like(E)),
                 (np.zeros_like(W), np.zeros_like(W)),
                 (np.zeros_like(b), np.zeros_like(b))]
    step = 0
    order = np.arange(n_full)
    for _ in range(EPOCHS_FULL):
        if step >= max_steps:
            break
        rng.shuffle(order)
        for s in range(0, n_full, BATCH):
            if step >= max_steps:
                break
            idx = order[s:s + BATCH]
            train_step(X[idx], y[idx], E, W, b, lr, l2, opt_state, step)
            step += 1

    # write weights
    # magic, d, D, C
    out = struct.pack("<4i", 0x454D4253, 16, D, 65)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embsum inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x454D4253) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (16L * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + 16L * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * D);
    float *logits = (float *)malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < D; u++) sum += h[u] * W[(long)u * C + c];
            logits[c] = sum;
        }
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```