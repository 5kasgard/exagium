```hypothesis
Family: label-fm (Factorization Machine for tabular classification).
Why: Adult income benefits from feature interactions (e.g., education×occupation, marital-status×relationship). 
FM captures all pairwise interactions via low-rank embeddings (k=8) with O(d·k) inference, 
adding minimal cost over linear model. Beats label-linear accuracy; matches label-mlp16 at lower latency.
Failure: If dev accuracy ≤0.8524 (label-linear) or Ir/item ≥1500 (2× label-linear), the interaction gain 
does not justify complexity.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Learns per-class linear weights and k-dim embeddings
for each (feature, value) pair. Scores: bias + linear + 0.5*sum_d((sum_j emb_d)^2 - sum_j emb_d^2).
Hyperparameter k chosen on last 20% tuning split, then retrained on all data. Adam lr=1e-3, L2=1e-4."""
import json, math, struct, sys

import numpy as np

K_CANDIDATES = (4, 8, 16, 32)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, params, d, k, C):
    n = X.shape[0]
    scores = np.zeros((n, C), np.float32)
    for c in range(C):
        scores[:, c] = params['bias'][c]
        for j in range(d):
            scores[:, c] += params['W'][c, j, X[:, j]]
        emb_sum = np.zeros((n, k), np.float32)
        emb_sum2 = np.zeros((n, k), np.float32)
        for j in range(d):
            v = params['V'][c, j, X[:, j]]
            emb_sum += v
            emb_sum2 += v * v
        scores[:, c] += 0.5 * np.sum(emb_sum * emb_sum - emb_sum2, axis=1)
    return scores


def train_fm(X, y, d, k, C, rng):
    n = X.shape[0]
    bias = np.zeros(C, np.float32)
    W = np.zeros((C, d, 256), np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, k)) * math.sqrt(2.0 / k)).astype(np.float32)
    params_list = [bias, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params_list]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            scores = fm_forward(xb, {'bias': bias, 'W': W, 'V': V}, d, k, C)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(len(yb)), yb] -= 1.0
            grad_scores /= len(yb)
            grad_bias = grad_scores.sum(axis=0)
            grad_W = np.zeros_like(W)
            for c in range(C):
                for j in range(d):
                    np.add.at(grad_W[c, j], xb[:, j], grad_scores[:, c])
            grad_V = np.zeros_like(V)
            for c in range(C):
                S = np.zeros((len(xb), k), np.float32)
                for j in range(d):
                    S += V[c, j, xb[:, j]]
                for j in range(d):
                    vals = xb[:, j]
                    V_j = V[c, j, vals]
                    g = grad_scores[:, c][:, None]
                    np.add.at(grad_V[c, j], vals, g * (S - V_j))
            grads = [grad_bias, grad_W, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params_list, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return {'bias': bias, 'W': W, 'V': V}


def evaluate(X, y, params, d, k, C):
    scores = fm_forward(X, params, d, k, C)
    return np.mean(np.argmax(scores, axis=1) == y)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    n = len(X_all)
    split = int(n * 0.8)
    X_tr, X_tu = X_all[:split], X_all[split:]
    y_tr, y_tu = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_k, best_acc = 8, -1.0
    for k in K_CANDIDATES:
        params = train_fm(X_tr, y_tr, d, k, C, rng)
        acc = evaluate(X_tu, y_tu, params, d, k, C)
        print(f"k={k} tune_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_k = acc, k
    print(f"best_k={best_k} retraining")
    rng = np.random.default_rng(SEED)
    params = train_fm(X_all, y_all, d, best_k, C, rng)
    out = struct.pack("<4i", 0x464D3131, d, best_k, C)
    out += params['bias'].astype("<f4").tobytes()
    out += params['W'].astype("<f4").tobytes()
    out += params['V'].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0/1).
 * Weights format: magic(0x464D3131), d, k, C; bias[C]; W[C][d][256]; V[C][d][256][k] (all float32 LE). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || d > 256 || k < 1 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)C * d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    const float *V = W + (long)C * d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2], S[2][64], S2[2][64];
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        scores[0] = bias[0]; scores[1] = bias[1];
        for (int c = 0; c < 2; c++) for (int dim = 0; dim < k; dim++) S[c][dim] = S2[c][dim] = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < 2; c++) {
                scores[c] += W[((long)c * d + j) * 256 + v];
                const float *emb = V + (((long)c * d + j) * 256 + v) * k;
                for (int dim = 0; dim < k; dim++) {
                    float e = emb[dim];
                    S[c][dim] += e;
                    S2[c][dim] += e * e;
                }
            }
        }
        for (int c = 0; c < 2; c++) {
            for (int dim = 0; dim < k; dim++) {
                scores[c] += 0.5f * (S[c][dim] * S[c][dim] - S2[c][dim]);
            }
        }
        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```