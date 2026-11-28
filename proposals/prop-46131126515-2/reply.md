```hypothesis
Family: label-fm (Factorization Machine with factorized bigram interactions on last 4 positions).
Why: Next-character prediction relies heavily on local bigrams (e.g., "th"→"e"). The linear model uses only unigram features; label-fm adds rank-16 factorized interactions for the last 4 adjacent byte pairs, capturing bigram statistics with only ~30K extra parameters. Inference stays fast (O(d*C + pairs*K*C)) by computing interactions on the fly from small embedding tables.
Failure mode: If dev accuracy ≤ 0.3375 (label-linear), the factorized bigrams do not add sufficient signal over unigrams for this 16-byte context.
```
```python
"""label-fm: Linear unigram + factorized bigram interactions (last 4 adjacent pairs).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D4231, d, H=0, C, K, num_pairs=4; float32 T[d][256][C], b1[C], U[4][256][K], V[4][256][K], Wf[C][K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 16                  # factor dimension
NUM_PAIRS = 4           # last 4 adjacent pairs: (11,12), (12,13), (13,14), (14,15)
START_POS = 16 - NUM_PAIRS - 1  # 11

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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Build per-position one-hot encoders (categorical -> 256-dim one-hot)
    enc = [np.eye(256, dtype=np.float32) for _ in range(d)]
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1)  # (n, d*256)
    n, k = Z.shape  # k = d*256 = 4096

    # Parameters
    W1 = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    U = rng.normal(0, 0.01, (NUM_PAIRS, 256, K)).astype(np.float32)
    V = rng.normal(0, 0.01, (NUM_PAIRS, 256, K)).astype(np.float32)
    Wf = rng.normal(0, 0.01, (C, K)).astype(np.float32)

    params = [W1, b1, U, V, Wf]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            xb = X[idx]  # (B, d)

            # Linear logits
            lin = z @ W1 + b1  # (B, C)

            # Bigram logits
            big = np.zeros((len(yy), C), np.float32)
            for p in range(NUM_PAIRS):
                pu, pv = START_POS + p, START_POS + p + 1
                u_vals = U[p, xb[:, pu]]          # (B, K)
                v_vals = V[p, xb[:, pv]]          # (B, K)
                temp = u_vals * v_vals            # (B, K)
                big += temp @ Wf.T                # (B, C)

            logits = lin + big
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # (B, C)

            # Gradients
            grad_W1 = z.T @ g + L2 * W1
            grad_b1 = g.sum(axis=0)
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_Wf = np.zeros_like(Wf)

            for p in range(NUM_PAIRS):
                pu, pv = START_POS + p, START_POS + p + 1
                u_vals = U[p, xb[:, pu]]      # (B, K)
                v_vals = V[p, xb[:, pv]]      # (B, K)
                temp = u_vals * v_vals        # (B, K)

                grad_Wf += g.T @ temp         # (C, K)
                factor = g @ Wf               # (B, K)
                u_idx = xb[:, pu]
                v_idx = xb[:, pv]
                np.add.at(grad_U[p], u_idx, v_vals * factor)
                np.add.at(grad_V[p], v_idx, u_vals * factor)

            grad_Wf += L2 * Wf
            grad_U += L2 * U
            grad_V += L2 * V

            grads = [grad_W1, grad_b1, grad_U, grad_V, grad_Wf]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export per-position lookup tables T[j][byte][c] = W1 slice
    T = []
    off = 0
    for j in range(d):
        T.append(W1[off:off + 256].copy())  # (256, C)
        off += 256

    magic = 0x464D4231  # 'FMB1'
    hdr = struct.pack("<6i", magic, d, 0, C, K, NUM_PAIRS)
    out = hdr
    out += np.stack(T).astype("<f4").tobytes()   # (d, 256, C)
    out += b1.astype("<f4").tobytes()            # (C,)
    out += U.astype("<f4").tobytes()             # (NUM_PAIRS, 256, K)
    out += V.astype("<f4").tobytes()             # (NUM_PAIRS, 256, K)
    out += Wf.astype("<f4").tobytes()            # (C, K)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Linear unigram + factorized bigram interactions (last 4 adjacent pairs).
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 24 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], K = hd[4], NP = hd[5];
    if (H != 0 || d != 16 || C != 65 || K != 16 || NP != 4) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 24L
        + (long)d * 256 * C * 4       // T
        + (long)C * 4                 // b1
        + (long)NP * 256 * K * 4 * 2  // U, V
        + (long)C * K * 4;            // Wf
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T   = (const float *)(hd + 6);
    const float *b1  = T + (long)d * 256 * C;
    const float *U   = b1 + C;
    const float *V   = U + (long)NP * 256 * K;
    const float *Wf  = V + (long)NP * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *temp   = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tj[c];
        }

        for (int p = 0; p < NP; p++) {
            int pu = 11 + p, pv = 12 + p;
            const float *up = U + ((long)p * 256 + xi[pu]) * K;
            const float *vp = V + ((long)p * 256 + xi[pv]) * K;
            for (int kk = 0; kk < K; kk++) temp[kk] = up[kk] * vp[kk];
            for (int c = 0; c < C; c++) {
                const float *wfc = Wf + (long)c * K;
                float sum = 0.0f;
                for (int kk = 0; kk < K; kk++) sum += temp[kk] * wfc[kk];
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