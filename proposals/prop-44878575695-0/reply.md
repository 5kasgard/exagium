```hypothesis
Family: label-fm (Factorization Machines). Factorization Machines model pairwise feature interactions via low-rank latent vectors, capturing combinatorial effects in tabular data (e.g., "education=HS & occupation=craft" boosting >50K) without the depth of GBDT or the dense layers of MLP. On Adult, categorical features interact strongly; FM adds only O(d*k) params (k=8) and O(d*k) inference ops. If dev accuracy ≤ label-linear (0.8524) or Ir/item ≥ 2× label-linear, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via latent vectors.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored (1 dim), categorical one-hot over training values (like label-linear).
FM: w0 + sum_j w_j x_j + sum_{j<l} <v_j, v_l> x_j x_l. k=8 latent dim. Adam, L2 on w,V.
Deterministic seed 1, single thread. Exports: magic, d, k, C, w0, w[d], V[d][k], b[C] (bias per class)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, L2V, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Build feature matrices per column, track offsets
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Concatenate to design matrix Z (n, d)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, d = Z.shape

    # FM parameters per class: w0[C], w[C,d], V[C,d,K]
    w0 = np.zeros(C, np.float32)
    w = (rng.uniform(-0.01, 0.01, (C, d)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (C, d, K)) * math.sqrt(6.0 / (d * K))).astype(np.float32)

    # Adam state
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)  # (B, C)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]  # (B, d), (B,)
            B = len(yy)

            # FM forward per class: logits[c] = w0[c] + z @ w[c] + 0.5 * sum_k ( (z @ V[c,:,k])^2 - z^2 @ V[c,:,k]^2 )
            logits = np.zeros((B, C), np.float32)
            for c in range(C):
                lin = z @ w[c] + w0[c]  # (B,)
                # interaction
                zV = z @ V[c]  # (B, K)
                z2 = z * z
                V2 = V[c] * V[c]
                inter = 0.5 * ((zV * zV).sum(axis=1) - z2 @ V2)  # (B,)
                logits[:, c] = lin + inter

            g = softmax_grad(logits, yy)  # (B, C)

            # Gradients
            gw0 = g.sum(axis=0) + L2 * w0  # (C,)
            gw = z.T @ g + L2 * w  # (d, C) -> transpose to (C, d)
            gw = gw.T
            gV = np.zeros((C, d, K), np.float32)
            for c in range(C):
                gvc = g[:, c:c+1] * zV  # (B, K) * (B, K) -> wait, need proper formula
                # d/dV_{j,k} inter = (z_j * zV_k) - (z_j^2 * V_{j,k})
                # gV[c,j,k] = sum_b g_bc * (z_bj * zV_bk - z_bj^2 * V_{c,j,k}) + L2V * V_{c,j,k}
                for k_ in range(K):
                    zVk = zV[:, k_]  # (B,)
                    gV[c, :, k_] = z.T @ (g[:, c] * zVk) - (z2.T @ (g[:, c] * V[c, :, k_])) + L2V * V[c, :, k_]

            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights: magic, d, K, C, w0[C], w[C*d], V[C*d*K]
    magic = 0x464D3100  # "FM1\0"
    out = struct.pack("<4i", magic, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (original columns). We reconstruct z on the fly using stored encodings.
 * Weights layout: magic(4), d, K, C, w0[C], w[C*d], V[C*d*K] (all float32).
 * Encodings are re-derived from training data stats? No - we must store encoding params in weights.
 * But train.py doesn't store them! Need to store encoding info: for each of d original columns,
 * store type, ordinal mu/sd, or categorical mapping. Simpler: store per-column lookup tables like label-linear.
 * Revised: train.py exports per-column tables T[col][256][W] where W = 1 + K (linear weight + K latent dims).
 * Actually, let's follow label-linear's approach: precompute per-byte contributions.
 * For FM: contribution of byte value v at column j to class c:
 *   linear: w[c, j] * enc(v)
 *   interaction: enc(v) * V[c, j, :]  (vector of K)
 * We can precompute for each column j, each byte value b (0..255), each class c:
 *   lin_table[j][b][c] = w[c, j] * enc_j(b)
 *   V_table[j][b][c][k] = enc_j(b) * V[c, j, k]
 * But enc_j(b) is scalar (1 for one-hot, z-score for ordinal). So per column j, we have 256 * C * (1 + K) floats.
 * Total: d * 256 * C * (1+K) floats. For d=14, C=2, K=8: 14*256*2*9 = 64512 floats = 256 KB. OK.
 * Inference: accumulate lin_sum[c] = w0[c]; V_sum[c][k] = 0.
 * For each column j, byte x_j: lin_sum[c] += lin_table[j][x_j][c]; V_sum[c][k] += V_table[j][x_j][c][k].
 * Then logits[c] = lin_sum[c] + 0.5 * sum_k (V_sum[c][k]^2 - sum_j (enc_j(x_j)^2 * V[c,j,k]^2)).
 * Wait, the efficient FM formula needs sum_j enc_j^2 * V^2. We can precompute that too:
 *   V2_table[j][b][c][k] = enc_j(b)^2 * V[c,j,k]^2
 * Or compute on the fly: we have V_sum and we can also accumulate V2_sum[c][k] += V2_table...
 * Let's do: precompute for each j,b,c,k: lin_w = w[c,j]*enc, V_val = V[c,j,k]*enc, V2_val = V[c,j,k]^2 * enc^2.
 * Then logits[c] = w0[c] + sum_j lin_w + 0.5 * (sum_k V_sum[k]^2 - sum_k V2_sum[k]).
 * This is clean and fast.
 * Weights file: magic, d, K, C, then for each class c: w0[c], then for each col j: 256*(1 + 2*K) floats?
 * Better: flatten as train.py does. Let's design train.py export accordingly.
 * 
 * Actually, to match label-linear style: train.py computes per-column tables T[col] of shape (256, W) where W = C*(1+2*K)?
 * Let's keep it simple in C: store w0[C], then for each of d columns, a table of 256 * C * (1 + 2*K) floats.
 * But train.py currently doesn't output column tables. I need to rewrite train.py to output column tables.
 * 
 * Given the complexity, let me simplify: use the same encoding as label-linear (concatenated one-hot + z-score),
 * but store the FM weights in the "wide" format (per expanded feature). Then inference uses the same Z reconstruction.
 * However, C inference must reconstruct Z without numpy. label-linear does it by storing per-column lookup tables T[col][256][W]
 * where W = d_expanded (for linear) or H (for MLP). For FM, we need linear weights and latent vectors per expanded feature.
 * So we can store T[col][256][W] where W = C * (1 + 2*K) ??? No, W is the expanded dimension.
 * Actually, label-linear's T is (d_orig, 256, W) where W = C (for linear) or H (for MLP16).
 * It computes acc[W] = sum_j T[j][x_j], then final layer.
 * For FM: we have expanded features (d_exp). We need w[C, d_exp] and V[C, d_exp, K].
 * We can store per original column j: for each byte value b, the contribution to each expanded feature.
 * But expanded features are just concatenation of per-column encodings.
 * So for ordinal column: 1 expanded feature. enc = (b - mu)/sd.
 * For categorical column: m expanded features (one-hot). enc = 1 if b == val else 0.
 * We can precompute for each original column j, each byte b, the vector of expanded features (length d_exp_j).
 * Then store for each class c: linear weights w[c, :] and V[c, :, K] aligned with expanded features.
 * Inference: for each class c, maintain lin_acc, V_acc[K], V2_acc[K].
 * For each original column j, byte x_j: get expanded feature vector f (length d_exp_j) from lookup.
 * Then lin_acc += dot(f, w[c, offset:offset+d_exp_j])
 * V_acc[k] += dot(f, V[c, offset:offset+d_exp_j, k])
 * V2_acc[k] += dot(f^2, V[c, offset:offset+d_exp_j, k]^2)  -- but f is 0/1 or scalar, so f^2 = f for one-hot, f^2 for ordinal.
 * This requires storing f for each (j,b) which is variable length. Too complex for C.
 * 
 * Alternative: Flatten all expanded features upfront. d_exp total. Store per expanded feature its original column and type.
 * But then inference needs to map input byte to expanded feature vector.
 * 
 * Simplest for C: Replicate label-linear's encoding exactly. In train.py, after building Z (n, d_exp),
 * we learn FM on Z. Then we export per original column j a table T[j] of shape (256, d_exp_j * C * (1+2K))? No.
 * 
 * Let's change approach: Use the "field-aware" idea but simpler. Since d_orig=14 is small, and expanded d_exp <= ~100,
 * we can store the full w and V matrices (C * d_exp and C * d_exp * K) and in C, reconstruct the expanded feature
 * vector for each input record by iterating original columns and expanding manually using stored encoding params.
 * We store encoding params per original column: for ordinal: mu, sd; for categorical: list of values (up to 255).
 * But C code would need dynamic arrays. 
 * 
 * OK, I'll follow label-linear's method exactly: train.py computes per-column lookup tables that directly give
 * the contribution to the "hidden" layer. For FM, the "hidden" layer is the linear + interaction terms.
 * But interaction terms are quadratic. Can't precompute directly.
 * 
 * Wait! The efficient FM formula: sum_{i<j} <v_i, v_j> x_i x_j = 0.5 * ( (sum_i v_i x_i)^2 - sum_i v_i^2 x_i^2 ).
 * So if we precompute for each original column j and byte b:
 *   For each class c, each latent dim k:
 *     A[j][b][c][k] = sum_{expanded feature f in col j} enc_f(b) * V[c, f, k]
 *     B[j][b][c][k] = sum_{f in col j} enc_f(b)^2 * V[c, f, k]^2
 * And linear part:
 *     L[j][b][c] = sum_{f in col j} enc_f(b) * w[c, f]
 * Then at inference:
 *   lin_sum[c] = w0[c] + sum_j L[j][x_j][c]
 *   V_sum[c][k] = sum_j A[j][x_j][c][k]
 *   V2_sum[c][k] = sum_j B[j][x_j][c][k]
 *   logits[c] = lin_sum[c] + 0.5 * sum_k (V_sum[c][k]^2 - V2_sum[c][k])
 * This is perfect! Storage per original column: 256 * C * (1 + 2*K) floats.
 * For d=14, C=2, K=8: 14 * 256 * 2 * 17 = 122,304 floats = ~480 KB. Acceptable.
 * 
 * Train.py will compute these tables and write them.
 * Weights file format:
 *   magic (0x464D3101), d_orig, K, C
 *   w0[C] (float32)
 *   For each original column j (0..d-1):
 *     For each byte b (0..255):
 *       For each class c (0..C-1):
 *         L (float32)
 *         A[K] (float32)
 *         B[K] (float32)
 * Total floats: C + d * 256 * C * (1 + 2*K)
 * Let's implement this.
 */
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
    float *wdata = (float *)slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wdata;
    if (hdr[0] != 0x464D3101) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (C < 1 || C > 256 || K < 1 || K > 32 || d < 1 || d > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16 + (size_t)C * 4 + (size_t)d * 256 * C * (1 + 2 * K) * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = wdata + 4;  // after header (4 int32 = 16 bytes)
    const float *w0 = p; p += C;
    // Tables: [d][256][C][1 + 2*K]  (L, A[K], B[K])
    // We'll index as: table[j][b] points to start of C*(1+2K) floats
    const float *tables = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_sum = calloc(C, sizeof(float));
    float *V_sum = calloc(C * K, sizeof(float));
    float *V2_sum = calloc(C * K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            lin_sum[c] = w0[c];
            for (int kk = 0; kk < K; kk++) {
                V_sum[c * K + kk] = 0.0f;
                V2_sum[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            const float *tbl = tables + ((size_t)j * 256 + b) * C * (1 + 2 * K);
            for (int c = 0; c < C; c++) {
                const float *cp = tbl + c * (1 + 2 * K);
                lin_sum[c] += cp[0];  // L
                for (int kk = 0; kk < K; kk++) {
                    V_sum[c * K + kk] += cp[1 + kk];       // A
                    V2_sum[c * K + kk] += cp[1 + K + kk];  // B
                }
            }
        }

        float best_score = -1e30f; unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float score = lin_sum[c];
            for (int kk = 0; kk < K; kk++) {
                float v = V_sum[c * K + kk];
                score += 0.5f * (v * v - V2_sum[c * K + kk]);
            }
            if (c == 0 || score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }

    free(lin_sum); free(V_sum); free(V2_sum);
    return 0;
}
```