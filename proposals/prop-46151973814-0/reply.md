```hypothesis
Family: label-fm (Factorization Machine). Adult income classification benefits from pairwise feature interactions (e.g., education × occupation, marital-status × relationship) which linear models miss. A rank-8 FM adds only ~2k parameters over linear, keeping inference fast (O(d·k) per item). If dev accuracy ≤ 0.8524 (label-linear) or CPU cost > 2× label-linear, the family fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored (1 feature), categorical -> one-hot over training values.
FM: w0[c] + sum_i x_i W[i,c] + sum_f ( (sum_i x_i V[i,f,c])^2 - sum_i (x_i V[i,f,c])^2 ) / 2
Exported: magic, d, k, C, D; per-byte lookup (feat_idx int16, feat_val float32); w0[C]; W[D,C]; V[D,k,C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on last 20% of train.in)
K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 1e-2, 1e-4, 256, 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Split for tuning (last 20% in file order)
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    # Build feature mapping from training portion only
    feat_idx = np.full((d, 256), -1, dtype=np.int32)
    feat_val = np.zeros((d, 256), dtype=np.float32)
    D = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X_tr[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            feat_idx[j, :] = D
            feat_val[j, :] = (np.arange(256) - mu) / sd
            D += 1
        else:
            vals = np.unique(X_tr[:, j])
            for val in vals:
                feat_idx[j, val] = D
                feat_val[j, val] = 1.0
                D += 1
            # unseen (255 and others) remain -1, 0.0

    # Build dense feature matrices
    def build_Z(Xpart):
        Z = np.zeros((len(Xpart), D), dtype=np.float32)
        for j in range(d):
            idx = feat_idx[j, Xpart[:, j]]
            val = feat_val[j, Xpart[:, j]]
            mask = idx >= 0
            Z[mask, idx[mask]] = val[mask]
        return Z

    Z_tr = build_Z(X_tr)
    Z_va = build_Z(X_va)

    # Initialize parameters
    rng = np.random.default_rng(SEED)
    scale = math.sqrt(6.0 / (D + C))
    w0 = np.zeros(C, np.float32)
    W = (rng.uniform(-scale, scale, (D, C))).astype(np.float32)
    V = (rng.uniform(-scale, scale, (D, K, C))).astype(np.float32)

    # Adam state
    params = [w0, W, V]
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(Z):
        # Z: (n, D)
        # scores: (n, C)
        lin = Z @ W + w0  # (n, C)
        # Interaction: for each f, c: (sum_i z_i V_ifc)^2 - sum_i (z_i V_ifc)^2
        # ZV: (n, K, C) = sum_i Z_ni * V_i
        ZV = np.einsum('ni,ifc->nfc', Z, V)  # (n, K, C)
        inter = 0.5 * (ZV ** 2 - np.einsum('ni,ifc->nfc', Z ** 2, V ** 2)).sum(axis=1)  # (n, C)
        return lin + inter

    def loss_grad(Z, y):
        n = Z.shape[0]
        scores = forward(Z)  # (n, C)
        # softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (n, C)
        probs[np.arange(n), y] -= 1.0
        probs /= n  # (n, C) gradient wrt scores

        # grad w0
        gw0 = probs.sum(axis=0)  # (C,)
        # grad W: Z.T @ probs
        gW = Z.T @ probs  # (D, C)
        # grad V: for each f,c: sum_n probs_nc * (ZV_nfc * Z_ni - (Z_n^2 * V_ifc))
        ZV = np.einsum('ni,ifc->nfc', Z, V)  # (n, K, C)
        # term1: probs[n,c] * ZV[n,f,c] * Z[n,i] -> sum_n
        # term2: probs[n,c] * Z[n,i]^2 * V[i,f,c]
        # We can compute as:
        # gV_ifc = sum_n probs_nc * (ZV_nfc * Z_ni - Z_ni^2 * V_ifc)
        # = sum_n probs_nc * ZV_nfc * Z_ni - V_ifc * sum_n probs_nc * Z_ni^2
        # Let A_ifc = sum_n probs_nc * ZV_nfc * Z_ni
        # Let B_ic = sum_n probs_nc * Z_ni^2
        # gV_ifc = A_ifc - V_ifc * B_ic
        A = np.einsum('nfc,ni->ifc', probs * ZV, Z)  # (D, K, C) wait: probs (n,C), ZV (n,K,C) -> probs_nc * ZV_nfc = (n,K,C)
        # Actually: probs (n,C)[:, None, :] * ZV (n,K,C) -> (n,K,C)
        # Then einsum 'nkc,ni->ikc'? Let's do carefully.
        # probs: (n, C), ZV: (n, K, C)
        # We want for each i,f,c: sum_n probs[n,c] * ZV[n,f,c] * Z[n,i]
        # So: (probs[:, :, None] * ZV[:, :, :]) -> (n, K, C) but probs needs to broadcast over K.
        # probs_nc * ZV_nfc -> (n, K, C)
        P_ZV = probs[:, None, :] * ZV  # (n, K, C)
        A = np.einsum('nkc,ni->ikc', P_ZV, Z)  # (D, K, C)
        B = np.einsum('nc,ni->ic', probs, Z ** 2)  # (D, C)
        gV = A - V * B[:, None, :]  # (D, K, C)

        # L2 regularization
        gW += L2 * W
        gV += L2 * V
        return [gw0, gW, gV]

    best_acc = -1
    best_params = None

    for epoch in range(EPOCHS):
        order = rng.permutation(len(Z_tr))
        for s in range(0, len(Z_tr), BATCH):
            idx = order[s:s + BATCH]
            Z_batch = Z_tr[idx]
            y_batch = y_tr[idx]
            grads = loss_grad(Z_batch, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        scores_va = forward(Z_va)
        pred_va = scores_va.argmax(axis=1)
        acc = (pred_va == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]

    # Retrain on full data with best params (or just use best_params from early stopping)
    # For simplicity, we use the best params found during training on train_tune.
    # But contract says: choose settings on tuning split, then train on all train.
    # We'll do a final short training on full data starting from best_params.
    w0, W, V = best_params
    params = [w0, W, V]
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0

    Z_full = build_Z(X)
    for epoch in range(10):  # few epochs on full data
        order = rng.permutation(len(Z_full))
        for s in range(0, len(Z_full), BATCH):
            idx = order[s:s + BATCH]
            Z_batch = Z_full[idx]
            y_batch = y[idx]
            grads = loss_grad(Z_batch, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    w0, W, V = params

    # Export weights
    # Magic: 0x464D3130 ("FM10")
    # Header: magic, d, K, C, D
    header = struct.pack("<5i", 0x464D3130, d, K, C, D)
    # Per-byte lookup: for each j in 0..d-1, each v in 0..255: int16 feat_idx, float32 feat_val
    lookup = b""
    for j in range(d):
        for v in range(256):
            lookup += struct.pack("<hf", int(feat_idx[j, v]), feat_val[j, v])
    # w0[C]
    w0_bytes = w0.astype("<f4").tobytes()
    # W[D, C] row-major
    W_bytes = W.astype("<f4").tobytes()
    # V[D, K, C] -> flatten as D * K * C
    V_bytes = V.astype("<f4").tobytes()

    out = header + lookup + w0_bytes + W_bytes + V_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax over C).
 * Weights format: magic(0x464D3130), d, k, C, D; lookup[d][256]{int16 idx, float val}; w0[C]; W[D][C]; V[D][k][C] */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3], D = hd[4];
    if (C < 1 || C > 256 || K < 1 || K > 64 || D < 1 || D > 10000) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected size: header(20) + lookup(d*256*6) + w0(C*4) + W(D*C*4) + V(D*K*C*4)
    long expected = 20L + (long)d * 256 * 6 + (long)C * 4 + (long)D * C * 4 + (long)D * K * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    // Pointers
    unsigned char *p = wb + 20;
    int16_t (*feat_idx)[256] = (int16_t (*)[256])p; p += (long)d * 256 * 2;
    float (*feat_val)[256] = (float (*)[256])p; p += (long)d * 256 * 4;
    float *w0 = (float *)p; p += (long)C * 4;
    float *W = (float *)p; p += (long)D * C * 4;
    float *V = (float *)p; // D*K*C

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers
    float *scores = (float *)malloc(sizeof(float) * C);
    float *ZV = (float *)malloc(sizeof(float) * K * C); // accumulator for sum_i z_i V_ifc
    float *ZV2 = (float *)malloc(sizeof(float) * K * C); // accumulator for sum_i z_i^2 V_ifc^2

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize scores with w0
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        // Clear interaction accumulators
        for (int f = 0; f < K; f++) for (int c = 0; c < C; c++) {
            ZV[f * C + c] = 0.0f;
            ZV2[f * C + c] = 0.0f;
        }

        // Process each feature column
        for (int j = 0; j < d; j++) {
            int16_t idx = feat_idx[j][xi[j]];
            if (idx < 0) continue; // unseen categorical
            float val = feat_val[j][xi[j]];
            // Linear term
            for (int c = 0; c < C; c++) scores[c] += val * W[(long)idx * C + c];
            // Interaction accumulators
            for (int f = 0; f < K; f++) {
                for (int c = 0; c < C; c++) {
                    float v = V[((long)idx * K + f) * C + c] * val;
                    ZV[f * C + c] += v;
                    ZV2[f * C + c] += v * v;
                }
            }
        }

        // Add interaction terms
        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) {
                scores[c] += 0.5f * (ZV[f * C + c] * ZV[f * C + c] - ZV2[f * C + c]);
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(ZV); free(ZV2); free(wb); if (x) free(x);
    return 0;
}
```