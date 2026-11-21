```hypothesis
Family: label-fm (Factorization Machine for multiclass sequence labeling).
Why: Character prediction relies heavily on pairwise interactions between recent context positions (e.g., "t"+"h" → "e"). A Factorization Machine models all pairwise feature interactions via low-rank factors, capturing such n-gram patterns with O(k) parameters per feature instead of O(n²), while remaining a simple linear+dot-product computation at inference.
Failure signal: If the optimal factor rank k exceeds 16 (causing overfitting or inference cost >2× label-linear) or if validation accuracy plateaus below label-linear's 0.3375, the family is not suitable for this task.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, C, k, n_feat; int16 feat_id[d*256];
float32 W[n_feat*C], V[C*n_feat*k], V_sq[C*n_feat]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-2, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8, 16]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t. logits: (P - Y) / batch."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def fm_forward(W, V, V_sq, X_ids, C, k):
    """X_ids: (batch, d) feature indices (-1 for unseen). Return logits (batch, C)."""
    batch, d = X_ids.shape
    logits = np.zeros((batch, C), np.float32)
    # Linear term
    for c in range(C):
        logits[:, c] = W[X_ids.ravel(), c].reshape(batch, d).sum(axis=1)
    # Interaction term per class
    for c in range(C):
        S = np.zeros((batch, k), np.float32)
        sum_sq = np.zeros(batch, np.float32)
        for pos in range(d):
            fid = X_ids[:, pos]
            mask = fid >= 0
            if not mask.any():
                continue
            v = V[c, fid[mask]]          # (m, k)
            S[mask] += v
            sum_sq[mask] += V_sq[c, fid[mask]]
        norm_sq = (S * S).sum(axis=1)
        logits[:, c] += 0.5 * (norm_sq - sum_sq)
    return logits


def fm_backward(W, V, V_sq, X_ids, y, grad_logits, C, k, LR, L2):
    """SGD update for W, V (in-place). grad_logits = dL/dlogits (batch, C)."""
    batch, d = X_ids.shape
    # Linear gradients
    for c in range(C):
        g = grad_logits[:, c]  # (batch,)
        for pos in range(d):
            fid = X_ids[:, pos]
            mask = fid >= 0
            if mask.any():
                np.add.at(W, fid[mask], g[mask][:, None])
    W -= LR * (L2 * W)  # L2 regularization on W

    # Interaction gradients
    for c in range(C):
        g = grad_logits[:, c]  # (batch,)
        S = np.zeros((batch, k), np.float32)
        for pos in range(d):
            fid = X_ids[:, pos]
            mask = fid >= 0
            if mask.any():
                S[mask] += V[c, fid[mask]]
        # dL/dV = g * (S * dL/dnorm_sq - dL/dsum_sq) but dL/dnorm_sq = 0.5*g, dL/dsum_sq = -0.5*g
        # So for each active feature: dL/dV = 0.5 * g * (S - v)
        for pos in range(d):
            fid = X_ids[:, pos]
            mask = fid >= 0
            if not mask.any():
                continue
            v = V[c, fid[mask]]           # (m, k)
            S_m = S[mask]                 # (m, k)
            g_m = g[mask][:, None]        # (m, 1)
            V[c, fid[mask]] -= LR * (0.5 * g_m * (S_m - v) + L2 * v)


def train_fm(X_ids, y, C, k, n_feat, val_ids, val_y):
    """Train FM with Adam, return (W, V, V_sq)."""
    rng = np.random.default_rng(SEED)
    W = np.zeros((n_feat, C), np.float32)
    V = (rng.normal(0, 0.01, (C, n_feat, k)) * math.sqrt(2.0 / k)).astype(np.float32)
    V_sq = (V * V).sum(axis=2).astype(np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    t = 0
    best_val = -1
    best_params = None

    n = len(X_ids)
    steps_per_epoch = max(1, n // BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    for step in range(max_steps):
        idx = rng.permutation(n)[:BATCH]
        xb, yb = X_ids[idx], y[idx]

        logits = fm_forward(W, V, V_sq, xb, C, k)
        grad_logits = softmax_grad(logits, yb)

        # Adam update
        t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        # Linear
        gW = np.zeros_like(W)
        for c in range(C):
            g = grad_logits[:, c]
            for pos in range(xb.shape[1]):
                fid = xb[:, pos]
                mask = fid >= 0
                if mask.any():
                    np.add.at(gW, fid[mask], g[mask][:, None])
        gW += L2 * W
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * (gW * gW)
        W -= LR * (mW / (1 - beta1 ** t)) / (np.sqrt(vW / (1 - beta2 ** t)) + eps)

        # Factors
        for c in range(C):
            g = grad_logits[:, c]
            S = np.zeros((BATCH, k), np.float32)
            for pos in range(xb.shape[1]):
                fid = xb[:, pos]
                mask = fid >= 0
                if mask.any():
                    S[mask] += V[c, fid[mask]]
            gV = np.zeros_like(V[c])
            for pos in range(xb.shape[1]):
                fid = xb[:, pos]
                mask = fid >= 0
                if not mask.any():
                    continue
                v = V[c, fid[mask]]
                S_m = S[mask]
                g_m = g[mask][:, None]
                np.add.at(gV, fid[mask], 0.5 * g_m * (S_m - v))
            gV += L2 * V[c]
            mV[c] = beta1 * mV[c] + (1 - beta1) * gV
            vV[c] = beta2 * vV[c] + (1 - beta2) * (gV * gV)
            V[c] -= LR * (mV[c] / (1 - beta1 ** t)) / (np.sqrt(vV[c] / (1 - beta2 ** t)) + eps)
        V_sq[:] = (V * V).sum(axis=2)

        # Validation check every epoch
        if (step + 1) % steps_per_epoch == 0:
            val_logits = fm_forward(W, V, V_sq, val_ids, C, k)
            val_pred = val_logits.argmax(axis=1)
            val_acc = (val_pred == val_y).mean()
            if val_acc > best_val:
                best_val = val_acc
                best_params = (W.copy(), V.copy(), V_sq.copy())

    if best_params is None:
        best_params = (W, V, V_sq)
    return best_params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16

    # Load data
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = len(X)

    # Build feature vocabulary per position (observed bytes only)
    feat_id = -np.ones((d, 256), dtype=np.int16)
    n_feat = 0
    for pos in range(d):
        vals = np.unique(X[:, pos])
        for v in vals:
            feat_id[pos, v] = n_feat
            n_feat += 1

    # Map X to feature indices
    X_ids = np.full((n, d), -1, dtype=np.int16)
    for pos in range(d):
        X_ids[:, pos] = feat_id[pos, X[:, pos]]

    # Split: last 20% for tuning
    split = n * 4 // 5
    tr_ids, val_ids = X_ids[:split], X_ids[split:]
    tr_y, val_y = y[:split], y[split:]

    # Tune k
    best_k, best_val = 4, -1
    for k in K_CANDIDATES:
        W, V, V_sq = train_fm(tr_ids, tr_y, C, k, n_feat, val_ids, val_y)
        val_logits = fm_forward(W, V, V_sq, val_ids, C, k)
        val_acc = (val_logits.argmax(axis=1) == val_y).mean()
        if val_acc > best_val:
            best_val, best_k = val_acc, k

    # Retrain on all data with best k
    W, V, V_sq = train_fm(X_ids, y, C, best_k, n_feat, X_ids, y)

    # Export weights
    magic = 0x464D3130  # "FM10"
    header = struct.pack("<5i", magic, d, C, best_k, n_feat)
    fid_bytes = feat_id.astype("<i2").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    V_bytes = V.astype("<f4").tobytes()
    Vsq_bytes = V_sq.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + fid_bytes + W_bytes + V_bytes + Vsq_bytes)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: one byte class index (argmax, first max wins).
 * Weights format: magic(0x464D3130), d, C, k, n_feat; int16 feat_id[d*256];
 * float32 W[n_feat*C], V[C*n_feat*k], V_sq[C*n_feat]. */
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
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], k = hd[3], n_feat = hd[4];
    if (d != 16 || C < 1 || C > 256 || k < 1 || k > 64 || n_feat < 1) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    int16_t *feat_id = (int16_t *)(hd + 5);
    long expect = 20L + (long)d * 256 * 2 + (long)n_feat * C * 4 + (long)C * n_feat * k * 4 + (long)C * n_feat * 4;
    if (nw != expect) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expect); return 3; }

    const float *W = (const float *)((const char *)feat_id + (long)d * 256 * 2);
    const float *V = W + (long)n_feat * C;
    const float *V_sq = V + (long)C * n_feat * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = (float *)malloc(sizeof(float) * k);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear term
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int p = 0; p < d; p++) {
            int fid = feat_id[p * 256 + xi[p]];
            if (fid >= 0) {
                const float *w = W + (long)fid * C;
                for (int c = 0; c < C; c++) scores[c] += w[c];
            }
        }

        // Interaction term per class
        for (int c = 0; c < C; c++) {
            memset(S, 0, sizeof(float) * k);
            float sum_sq = 0.0f;
            for (int p = 0; p < d; p++) {
                int fid = feat_id[p * 256 + xi[p]];
                if (fid < 0) continue;
                const float *v = V + ((long)c * n_feat + fid) * k;
                for (int f = 0; f < k; f++) S[f] += v[f];
                sum_sq += V_sq[(long)c * n_feat + fid];
            }
            float norm_sq = 0.0f;
            for (int f = 0; f < k; f++) norm_sq += S[f] * S[f];
            scores[c] += 0.5f * (norm_sq - sum_sq);
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(S); free(scores);
    return 0;
}
```