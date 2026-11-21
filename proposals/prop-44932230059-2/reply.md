```hypothesis
Family: label-lowrank. Factorizes the per-position per-byte weight matrices into shared low-rank embeddings: each position j has a 256×R matrix U_j, and classes share a 65×R matrix V. Score for class c is bias_c + Σ_j U_j[byte_j]^T V_c. This shares statistical strength across positions and classes, reducing parameters from 266k (linear) to 16×256×R + 65×R + 65. For R=16, ~67k parameters may generalize better on 24k samples. Inference sums 16 R-vectors then multiplies by V, costing ~16R + 65R multiply-adds. If dev accuracy does not exceed label-linear (0.3375) at similar or lower CPU cost, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_R = [8, 16, 32, 64]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, d, R, rng, epochs, batch_size, lr, l2):
    n = X.shape[0]
    U = rng.normal(0, 0.01, size=(d, 256, R)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(C, R)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    m_U, v_U = np.zeros_like(U), np.zeros_like(U)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]
            yb = y[idx]
            bsize = len(yb)
            sum_vec = np.zeros((bsize, R), dtype=np.float32)
            for j in range(d):
                sum_vec += U[j, xb[:, j]]
            scores = sum_vec @ V.T + bias
            max_score = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            probs /= bsize
            grad_V = probs.T @ sum_vec
            grad_bias = probs.sum(axis=0)
            grad_sum_vec = probs @ V
            grad_U = np.zeros_like(U)
            for j in range(d):
                np.add.at(grad_U[j], xb[:, j], grad_sum_vec)
            grad_U += l2 * U
            grad_V += l2 * V
            grad_bias += l2 * bias
            t += 1
            m_U = 0.9 * m_U + 0.1 * grad_U
            v_U = 0.999 * v_U + 0.001 * grad_U * grad_U
            U -= lr * (m_U / (1 - 0.9 ** t)) / (np.sqrt(v_U / (1 - 0.999 ** t)) + 1e-8)
            m_V = 0.9 * m_V + 0.1 * grad_V
            v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
            V -= lr * (m_V / (1 - 0.9 ** t)) / (np.sqrt(v_V / (1 - 0.999 ** t)) + 1e-8)
            m_b = 0.9 * m_b + 0.1 * grad_bias
            v_b = 0.999 * v_b + 0.001 * grad_bias * grad_bias
            bias -= lr * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
    return U, V, bias

def evaluate(X, y, C, d, U, V, bias):
    n = X.shape[0]
    R = U.shape[2]
    sum_vec = np.zeros((n, R), dtype=np.float32)
    for j in range(d):
        sum_vec += U[j, X[:, j]]
    scores = sum_vec @ V.T + bias
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    n_train = train_X.shape[0]
    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    rng = np.random.default_rng(SEED)
    best_R = CANDIDATE_R[0]
    best_acc = -1.0
    for R in CANDIDATE_R:
        rng_R = np.random.default_rng(SEED + R)
        U, V, bias = train_model(train_X, train_y, C, d, R, rng_R, epochs_tune, BATCH, LR, L2)
        acc = evaluate(tune_X, tune_y, C, d, U, V, bias)
        if acc > best_acc:
            best_acc = acc
            best_R = R
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_full = np.random.default_rng(SEED)
    U, V, bias = train_model(X, y, C, d, best_R, rng_full, epochs_full, BATCH, LR, L2)
    magic = 0x4C524B31
    out = struct.pack("<4i", magic, d, best_R, C)
    out += U.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x4C524B31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * R + (long)C * R + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *U = (const float *)(hd + 4);
    const float *V = U + (long)d * 256 * R;
    const float *bias = V + (long)C * R;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vec = (float *)calloc(R, sizeof(float));
    float *scores = (float *)malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int r = 0; r < R; r++) sum_vec[r] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) sum_vec[r] += u[r];
        }
        for (int c = 0; c < C; c++) {
            float acc = bias[c];
            const float *v = V + (long)c * R;
            for (int r = 0; r < R; r++) acc += sum_vec[r] * v[r];
            scores[c] = acc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```