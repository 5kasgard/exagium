```hypothesis
Family: label-fm (factorization machine on per-pixel embeddings). 
Why: Optdigits has 64 ordinal pixels; a degree-2 FM captures pairwise interactions with O(W·k) extra parameters (W=embedding size, k=latent dim), staying far cheaper than an MLP while modeling non-linearity. 
Failure signal: dev accuracy ≤ label-linear (0.9613) or latency ≥ label-mlp16 (776 ns/item).
```

```python
"""label-fm: Factorization Machine on per-pixel embeddings for label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, W, k, C; float32 T[d][256][W], b[C], u[C][W], V[C][W][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
W, K = 16, 4          # embedding size, FM latent dim (tuned on last 20% of train)

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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # --- encoding: ordinal z-score per pixel (same as label-linear) ---
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    # concatenated encoding matrix E_all [256*d, total_enc_dim] but we only need per-pixel 256 x enc_dim_j
    enc_dims = [e.shape[1] for e in enc]
    total_enc_dim = sum(enc_dims)

    # --- split for tuning (last 20% in file order) ---
    n = X.shape[0]
    split = int(0.8 * n)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)

    # --- build encoding vectors for all samples ---
    Z_enc = np.zeros((n, total_enc_dim), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        Z_enc[:, off:off+E.shape[1]] = E[X[:, j]]
        off += E.shape[1]

    # --- hyperparameter search on validation split (tiny grid) ---
    best_acc, best_cfg = -1, None
    for W_try in (8, 16, 32):
        for K_try in (2, 4, 8):
            # quick train on idx_tr, eval on idx_va
            W1 = (rng.uniform(-1, 1, (total_enc_dim, W_try)) * math.sqrt(6.0 / total_enc_dim)).astype(np.float32)
            b = np.zeros(C, np.float32)
            u = (rng.uniform(-1, 1, (C, W_try)) * math.sqrt(6.0 / W_try)).astype(np.float32)
            V = (rng.uniform(-1, 1, (C, W_try, K_try)) * math.sqrt(6.0 / W_try)).astype(np.float32)
            params = [W1, b, u, V]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            for _ in range(min(20, math.ceil(SAMPLES / len(idx_tr)))):  # few epochs for tuning
                order = rng.permutation(len(idx_tr))
                for s in range(0, len(idx_tr), BATCH):
                    idx = idx_tr[order[s:s+BATCH]]
                    z_enc = Z_enc[idx]
                    yy = y[idx]
                    z = z_enc @ W1                     # [B, W]
                    # scores: b + z u^T + ||z V_c^T||^2 per class
                    lin = z @ u.T + b                  # [B, C]
                    quad = np.zeros((len(idx), C), np.float32)
                    for c in range(C):
                        vz = z @ V[c].T                # [B, K]
                        quad[:, c] = np.sum(vz * vz, axis=1)
                    logits = lin + quad
                    # softmax cross-entropy gradient
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yy)), yy] -= 1.0
                    probs /= len(yy)
                    # gradients
                    g_u = z.T @ probs + L2 * u
                    g_b = probs.sum(0)
                    g_V = np.zeros_like(V)
                    for c in range(C):
                        vz = z @ V[c].T
                        g_V[c] = (z.T @ (probs[:, c:c+1] * 2 * vz)) + L2 * V[c]
                    g_W1 = z_enc.T @ (probs @ u + np.sum(2 * probs[:, :, None] * (z @ V.transpose(0,2,1)), axis=1)) + L2 * W1
                    grads = [g_W1, g_b, g_u, g_V]
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                        p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            # validation accuracy
            z_va = Z_enc[idx_va] @ W1
            lin_va = z_va @ u.T + b
            quad_va = np.zeros((len(idx_va), C), np.float32)
            for c in range(C):
                vz = z_va @ V[c].T
                quad_va[:, c] = np.sum(vz * vz, axis=1)
            pred = np.argmax(lin_va + quad_va, axis=1)
            acc = (pred == y[idx_va]).mean()
            if acc > best_acc:
                best_acc, best_cfg = acc, (W_try, K_try, W1.copy(), b.copy(), u.copy(), V.copy())
    W, K, W1, b, u, V = best_cfg

    # --- retrain on all data with best config ---
    params = [W1, b, u, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z_enc = Z_enc[idx]
            yy = y[idx]
            z = z_enc @ W1
            lin = z @ u.T + b
            quad = np.zeros((len(idx), C), np.float32)
            for c in range(C):
                vz = z @ V[c].T
                quad[:, c] = np.sum(vz * vz, axis=1)
            logits = lin + quad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)
            g_u = z.T @ probs + L2 * u
            g_b = probs.sum(0)
            g_V = np.zeros_like(V)
            for c in range(C):
                vz = z @ V[c].T
                g_V[c] = (z.T @ (probs[:, c:c+1] * 2 * vz)) + L2 * V[c]
            g_W1 = z_enc.T @ (probs @ u + np.sum(2 * probs[:, :, None] * (z @ V.transpose(0,2,1)), axis=1)) + L2 * W1
            grads = [g_W1, g_b, g_u, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # --- export per-pixel lookup tables T[d][256][W] = E_j @ W1[off_j:off_j+enc_dim_j, :] ---
    T = []
    off = 0
    for j, E in enumerate(enc):
        T.append((E.astype(np.float32) @ W1[off:off+E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T)  # [d, 256, W]

    # --- write weights.bin ---
    # magic, d, W, K, C
    out = struct.pack("<5i", 0x464D4C31, d, W, K, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += u.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights format: magic 0x464D4C31, int32 d, W, K, C; float32 T[d][256][W], b[C], u[C][W], V[C][W][K]. */
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
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], W = hd[2], K = hd[3], C = hd[4];
    if (C < 1 || C > 256 || W < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * W + C + (long)C * W + (long)C * W * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b = T + (long)d * 256 * W;
    const float *u = b + C;
    const float *V = u + (long)C * W;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * W);
    float *vz = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) z[w] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) z[w] += t[w];
        }
        for (int c = 0; c < C; c++) {
            float lin = b[c];
            for (int w = 0; w < W; w++) lin += z[w] * u[(long)c * W + w];
            float quad = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sum = 0.0f;
                for (int w = 0; w < W; w++) sum += z[w] * V[((long)c * W + w) * K + kk];
                quad += sum * sum;
            }
            scores[c] = lin + quad;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```