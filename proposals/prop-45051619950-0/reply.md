```hypothesis
Factorization Machine (FM) with per-feature-value embeddings captures pairwise interactions
that linear models miss, while staying cheaper than MLPs by using a low embedding dimension (k=8).
The adult dataset has many categorical features where interactions matter (e.g., education × occupation).
If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ label-linear, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 input bytes as a categorical value (0..255).
Learns: bias w0, linear weights W_lin[14,256], embeddings V[14,256,k].
Logit = w0 + sum_j W_lin[j, x_j] + sum_{i<j} <V[i,x_i], V[j,x_j]>.
Binary classification: output 1 if logit>0 else 0.
Hyperparameters (k, lr, l2, epochs) chosen on last 20% of train.in (file order),
then model retrained on all train.in.
Weights format: magic 0x464D4231, int32 d,k,C; float32 w0, W_lin[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split
GRID = {
    'k': [4, 8, 16],
    'lr': [0.01, 0.03, 0.1],
    'l2': [1e-4, 1e-3, 1e-2],
    'epochs': [30],
    'batch': [64],
    'seed': [1],
}

SEED = 1  # global seed for final training


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_fm(X, y, X_val, y_val, k, lr, l2, epochs, batch, seed):
    """Train FM on (X,y), validate on (X_val,y_val). Return (val_acc, params)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # Parameters
    w0 = np.float32(0.0)
    W_lin = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def compute_logits(Xb):
        """Xb: (B, d) uint8 -> logits (B,)"""
        B = Xb.shape[0]
        lin = w0 + np.sum(W_lin[np.arange(d), Xb.T], axis=0)  # (B,)
        # Interaction via sum_v^2 - sum_v_sq
        sum_v = np.zeros((B, k), dtype=np.float32)
        sum_v_sq = np.zeros((B, k), dtype=np.float32)
        for j in range(d):
            v = V[j, Xb[:, j]]  # (B, k)
            sum_v += v
            sum_v_sq += v * v
        inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
        return lin + inter

    def compute_grads(Xb, yb):
        """Return gradients for w0, W_lin, V on batch."""
        B = Xb.shape[0]
        logits = compute_logits(Xb)
        probs = sigmoid(logits)
        grad_logit = probs - yb  # (B,)

        # Gradient w0
        gw0 = np.mean(grad_logit)

        # Gradient W_lin
        gW = np.zeros_like(W_lin)
        for j in range(d):
            idx = Xb[:, j]
            np.add.at(gW[j], idx, grad_logit)
        gW /= B
        gW += l2 * W_lin

        # Gradient V
        gV = np.zeros_like(V)
        # Recompute sum_v for this batch
        sum_v = np.zeros((B, k), dtype=np.float32)
        for j in range(d):
            sum_v += V[j, Xb[:, j]]
        for j in range(d):
            v = V[j, Xb[:, j]]  # (B, k)
            grad_inter = grad_logit[:, None] * (sum_v - v)  # (B, k)
            np.add.at(gV[j], Xb[:, j], grad_inter)
        gV /= B
        gV += l2 * V

        return gw0, gW, gV

    # Training loop
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            gw0, gW, gV = compute_grads(Xb, yb)
            t += 1
            # Adam update
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * (gW * gW)
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * (gV * gV)
            # Bias correction
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            W_lin -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Validation accuracy
    val_logits = compute_logits(X_val)
    val_pred = (val_logits > 0).astype(np.int64)
    val_acc = np.mean(val_pred == y_val)
    return val_acc, (w0, W_lin, V)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "Only binary classification supported"

    n = X.shape[0]
    n_train = int(0.8 * n)
    # Tuning split: last 20% in file order
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    best_acc = -1.0
    best_params = None
    best_hp = None

    # Grid search
    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for epochs in GRID['epochs']:
                    for batch in GRID['batch']:
                        for seed in GRID['seed']:
                            acc, params = train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, batch, seed)
                            if acc > best_acc:
                                best_acc = acc
                                best_params = params
                                best_hp = (k, lr, l2, epochs, batch, seed)
    print(f"Best hyperparams: k={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, epochs={best_hp[3]}, batch={best_hp[4]}, seed={best_hp[5]}; val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    k, lr, l2, epochs, batch, seed = best_hp
    _, (w0, W_lin, V) = train_fm(X, y, X_val, y_val, k, lr, l2, epochs, batch, seed)

    # Save weights
    magic = 0x464D4231  # 'FMB1'
    out = struct.pack("<4i", magic, d, k, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: magic 0x464D4231, int32 d,k,C; float32 w0, W_lin[d*256], V[d*256*k].
 * Input record: d bytes. Output: 1 byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || k < 1 || nw != 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * k)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *p = (const float *)(hd + 4);
    const float w0 = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float logit = w0;
        // Linear part
        for (int j = 0; j < d; j++) {
            logit += W_lin[(long)j * 256 + xi[j]];
        }
        // Pairwise interactions
        for (int i1 = 0; i1 < d - 1; i1++) {
            const float *vi = V + ((long)i1 * 256 + xi[i1]) * k;
            for (int i2 = i1 + 1; i2 < d; i2++) {
                const float *vj = V + ((long)i2 * 256 + xi[i2]) * k;
                float dot = 0.0f;
                for (int l = 0; l < k; l++) dot += vi[l] * vj[l];
                logit += dot;
            }
        }
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```