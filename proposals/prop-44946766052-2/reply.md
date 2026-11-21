```hypothesis
Family: label-fm2 (Factorization Machine degree-2 with low-rank bigram interactions).
Why: Next-character prediction depends heavily on local character transitions (bigrams). The linear model (label-linear, 0.3375 acc) sees each position independently. label-fm2 adds a rank-R interaction for each adjacent position pair (15 pairs for d=16), capturing bigram statistics with only ~35K extra parameters (R=4) and ~2x inference cost. This should significantly boost accuracy on text.
Failure signal: If dev accuracy ≤ 0.35, the low-rank bigrams aren't capturing useful signal beyond unigrams.
```
```python
"""label-fm2: Linear + low-rank bigram interactions for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format: magic 0x464D3231, d, R, C; float32 T_lin[d][256][C], b_lin[C];
for each adjacent pair j=0..d-2: A_j[256][R], B_j[256][R], C_j[C][R]."""
import json, math, struct, sys

import numpy as np

R = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    rng = np.random.default_rng(SEED)

    # Per-position observed values and one-hot encoders
    enc = []
    vals_list = []
    idx_list = []  # byte value -> index in vals (or -1)
    for j, t in enumerate(card["input"]["types"]):
        v = X_bytes[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
            vals_list.append(np.arange(256))
            idx_list.append(np.arange(256))
        else:
            vals = np.unique(v)
            vals_list.append(vals)
            V = len(vals)
            E = np.zeros((256, V), np.float32)
            E[vals, np.arange(V)] = 1.0
            enc.append(E)
            idx = np.full(256, -1, np.int32)
            idx[vals] = np.arange(V)
            idx_list.append(idx)

    # Linear params
    k = sum(E.shape[1] for E in enc)
    W_lin = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)

    # Bigram params per adjacent pair
    A_list, B_list, C_list = [], [], []
    for j in range(d - 1):
        Vj = enc[j].shape[1]
        Vj1 = enc[j + 1].shape[1]
        A = (rng.uniform(-1, 1, (Vj, R)) * math.sqrt(6.0 / Vj)).astype(np.float32)
        B = (rng.uniform(-1, 1, (Vj1, R)) * math.sqrt(6.0 / Vj1)).astype(np.float32)
        Cmat = (rng.uniform(-1, 1, (C, R)) * math.sqrt(6.0 / C)).astype(np.float32)
        A_list.append(A)
        B_list.append(B)
        C_list.append(Cmat)

    # Adam state
    params = [W_lin, b_lin] + A_list + B_list + C_list
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute linear design matrix Z (n, k)
    Z = np.concatenate([E[X_bytes[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # For bigram forward: indices into A_j, B_j
    idx_arrays = [idx_list[j][X_bytes[:, j]] for j in range(d)]  # list of (n,) int32

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            Bsz = len(yy)

            # Linear scores
            scores = z @ W_lin + b_lin

            # Bigram scores
            for j in range(d - 1):
                ij = idx_arrays[j][idx]
                ij1 = idx_arrays[j + 1][idx]
                Avals = A_list[j][ij]
                Bvals = B_list[j][ij1]
                tmp = Avals * Bvals
                scores += tmp @ C_list[j].T

            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(Bsz), yy] -= 1.0
            dS = probs / Bsz

            # Linear grads
            dW_lin = z.T @ dS + L2 * W_lin
            db_lin = dS.sum(0)

            # Bigram grads
            dA_list = [np.zeros_like(A) for A in A_list]
            dB_list = [np.zeros_like(B) for B in B_list]
            dC_list = [np.zeros_like(C) for C in C_list]

            for j in range(d - 1):
                ij = idx_arrays[j][idx]
                ij1 = idx_arrays[j + 1][idx]
                A = A_list[j]
                B = B_list[j]
                Cmat = C_list[j]
                Avals = A[ij]
                Bvals = B[ij1]
                tmp = Avals * Bvals

                dC = tmp.T @ dS + L2 * Cmat
                dC_list[j] = dC

                dtmp = dS @ Cmat
                dAvals = dtmp * Bvals
                dBvals = dtmp * Avals

                np.add.at(dA_list[j], ij, dAvals)
                np.add.at(dB_list[j], ij1, dBvals)

            # Adam update
            grads = [dW_lin, db_lin] + dA_list + dB_list + dC_list
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: expand to 256-wide tables for direct byte indexing
    T_lin = []
    off = 0
    for E in enc:
        T_lin.append((E @ W_lin[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T_lin = np.stack(T_lin)  # (d, 256, C)

    A_tables = []
    B_tables = []
    for j in range(d - 1):
        A_tab = np.zeros((256, R), np.float32)
        B_tab = np.zeros((256, R), np.float32)
        A_tab[vals_list[j]] = A_list[j]
        B_tab[vals_list[j + 1]] = B_list[j]
        A_tables.append(A_tab)
        B_tables.append(B_tab)

    out = struct.pack("<4i", 0x464D3231, d, R, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    for j in range(d - 1):
        out += A_tables[j].astype("<f4").tobytes()
        out += B_tables[j].astype("<f4").tobytes()
        out += C_list[j].astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm2 inference: ./infer weights.bin in.rec N
 * Linear + rank-R bigram interactions for adjacent positions.
 * Weight format: magic 0x464D3231, d, R, C;
 * float32 T_lin[d][256][C], b_lin[C];
 * for each pair j=0..d-2: A_j[256][R], B_j[256][R], C_j[C][R]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1 || R > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 16
        + 4L * d * 256 * C
        + 4L * C
        + 4L * (d - 1) * (256 * R + 256 * R + C * R);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *b_lin = T_lin + (long)d * 256 * C;
    const float *pair_data = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *tmp = malloc(sizeof(float) * R);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }

        const float *pdata = pair_data;
        for (int j = 0; j < d - 1; j++) {
            const float *A = pdata; pdata += 256 * R;
            const float *B = pdata; pdata += 256 * R;
            const float *Cmat = pdata; pdata += C * R;

            const float *Arow = A + xi[j] * R;
            const float *Brow = B + xi[j + 1] * R;
            for (int r = 0; r < R; r++) tmp[r] = Arow[r] * Brow[r];

            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                const float *Crow = Cmat + c * R;
                for (int r = 0; r < R; r++) sum += tmp[r] * Crow[r];
                scores[c] += sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```