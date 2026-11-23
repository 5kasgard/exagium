```hypothesis
Factorization Machine (FM) with low-rank embeddings for pairwise feature interactions. The adult dataset has strong categorical interactions (e.g., education×occupation) that linear models miss. FM adds only O(k*d) parameters (k=4, d=14) and computes interactions in O(k*d) time via the (sum v)^2 - sum v^2 trick, so inference stays fast. We tune k on the last 20% of train.in, then retrain on all data. If dev accuracy is below 0.855, the model does not justify its complexity over label-linear.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 byte features as a categorical with 256 values.
Learns per-class linear weights and k-dimensional embeddings for pairwise interactions.
Tunes embedding size k on the last 20% of train.in, then retrains on all data."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS_TUNE = 30
EPOCHS_FULL = 100
CANDIDATE_K = [2, 4, 8]  # embedding sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(params, X, k):
    """Compute scores for both classes. X: (n, 14) uint8. Returns (n, 2) float32."""
    w0, w, v = params  # w0:(2,), w:(2,14,256), v:(2,14,256,k)
    n = X.shape[0]
    scores = np.zeros((n, 2), dtype=np.float32)
    for c in range(2):
        # linear part
        lin = w0[c] + np.sum(w[c, np.arange(14), X.T], axis=0)  # (n,)
        # interaction part
        sum_v = np.zeros((n, k), dtype=np.float32)
        sum_v_sq = np.zeros((n, k), dtype=np.float32)
        for j in range(14):
            emb = v[c, j, X[:, j], :]  # (n, k)
            sum_v += emb
            sum_v_sq += emb * emb
        inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)  # (n,)
        scores[:, c] = lin + inter
    return scores


def fm_loss_and_grad(params, X, y, k, l2):
    """Cross-entropy loss and gradients for a batch. Returns loss, grads."""
    w0, w, v = params
    n = X.shape[0]
    # forward
    scores = fm_forward(params, X, k)  # (n,2)
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)  # (n,2)
    # loss
    log_probs = np.log(probs[np.arange(n), y] + 1e-12)
    loss = -np.mean(log_probs)
    # L2 regularization
    loss += 0.5 * l2 * (np.sum(w0 * w0) + np.sum(w * w) + np.sum(v * v))
    # gradient of loss w.r.t scores
    dscores = probs.copy()
    dscores[np.arange(n), y] -= 1.0
    dscores /= n  # (n,2)
    # gradients
    gw0 = np.zeros(2, dtype=np.float32)
    gw = np.zeros_like(w)
    gv = np.zeros_like(v)
    for c in range(2):
        dc = dscores[:, c]  # (n,)
        gw0[c] = np.sum(dc) + l2 * w0[c]
        # linear weights gradient
        for j in range(14):
            vals = X[:, j]
            np.add.at(gw[c, j], vals, dc)
        gw[c] += l2 * w[c]
        # embedding gradients
        # precompute sum_v per class per factor for the batch
        sum_v = np.zeros((n, k), dtype=np.float32)
        for j in range(14):
            sum_v += v[c, j, X[:, j], :]
        for j in range(14):
            vals = X[:, j]
            emb = v[c, j, vals, :]  # (n, k)
            # dL/dv = dc[:, None] * (sum_v - emb)
            grad_emb = dc[:, None] * (sum_v - emb)  # (n, k)
            np.add.at(gv[c, j], vals, grad_emb)
        gv[c] += l2 * v[c]
    return loss, (gw0, gw, gv)


def adam_update(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    w0, w, v = params
    gw0, gw, gv = grads
    m_w0, m_w, m_v, v_w0, v_w, v_v = state
    t += 1
    # update moments
    m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
    m_w = beta1 * m_w + (1 - beta1) * gw
    m_v = beta1 * m_v + (1 - beta1) * gv
    v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
    v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
    v_v = beta2 * v_v + (1 - beta2) * (gv * gv)
    # bias correction
    m_w0_hat = m_w0 / (1 - beta1 ** t)
    m_w_hat = m_w / (1 - beta1 ** t)
    m_v_hat = m_v / (1 - beta1 ** t)
    v_w0_hat = v_w0 / (1 - beta2 ** t)
    v_w_hat = v_w / (1 - beta2 ** t)
    v_v_hat = v_v / (1 - beta2 ** t)
    # parameter update
    w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
    w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
    v -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)
    return (w0, w, v), (m_w0, m_w, m_v, v_w0, v_w, v_v), t


def train_fm(X, y, k, epochs, rng):
    """Train FM on given data, return params."""
    n = X.shape[0]
    # initialize parameters
    w0 = np.zeros(2, dtype=np.float32)
    w = (rng.uniform(-0.01, 0.01, (2, 14, 256)) * math.sqrt(6.0 / 256)).astype(np.float32)
    v = (rng.uniform(-0.01, 0.01, (2, 14, 256, k)) * math.sqrt(6.0 / 256)).astype(np.float32)
    params = (w0, w, v)
    # Adam state
    state = (np.zeros_like(w0), np.zeros_like(w), np.zeros_like(v),
             np.zeros_like(w0), np.zeros_like(w), np.zeros_like(v))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            loss, grads = fm_loss_and_grad(params, Xb, yb, k, L2)
            params, state, t = adam_update(params, grads, state, t, LR)
    return params


def evaluate(params, X, y, k):
    scores = fm_forward(params, X, k)
    pred = np.argmax(scores, axis=1)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = None, -1.0
    for k in CANDIDATE_K:
        params = train_fm(X_train, y_train, k, EPOCHS_TUNE, rng)
        acc = evaluate(params, X_val, y_val, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # retrain on full data with best_k
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_fm(X, y, best_k, EPOCHS_FULL, rng)

    # serialize weights
    w0, w, v = final_params
    magic = 0x464D31  # "FM1"
    d = 14
    C = 2
    header = struct.pack("<4i", magic, d, best_k, C)
    # flatten in order: for each class: w0, w (d*256), v (d*256*k)
    body = bytearray()
    for c in range(C):
        body += w0[c].astype("<f4").tobytes()
        body += w[c].astype("<f4").tobytes()
        body += v[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is 14 bytes (one per feature). Output is one byte: class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k < 1 || k > 8) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t floats_per_class = 1 + (size_t)d * 256 + (size_t)d * 256 * k;
    size_t total_floats = (size_t)C * floats_per_class;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += (size_t)C * d * 256;
    const float *v = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];
    float sum_v[8], sum_v_sq[8];  // k <= 8

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if (rec_len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        for (int c = 0; c < C; c++) {
            // linear part
            float score = w0[c];
            for (int j = 0; j < d; j++) {
                score += w[c * d * 256 + j * 256 + xi[j]];
            }
            // interaction part
            for (int f = 0; f < k; f++) {
                sum_v[f] = 0.0f;
                sum_v_sq[f] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                const float *emb = v + ((size_t)c * d * 256 * k) + (j * 256 + xi[j]) * k;
                for (int f = 0; f < k; f++) {
                    float val = emb[f];
                    sum_v[f] += val;
                    sum_v_sq[f] += val * val;
                }
            }
            for (int f = 0; f < k; f++) {
                score += 0.5f * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
            }
            scores[c] = score;
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```