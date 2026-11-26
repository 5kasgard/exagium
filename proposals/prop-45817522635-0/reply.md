```hypothesis
Factorization Machines (FM) model pairwise feature interactions via low-rank embeddings, which are known to be effective on tabular data with categorical features. The adult dataset has 14 mixed ordinal/categorical columns; a linear model misses interactions. FM adds O(d*K) parameters (d=14, K=4) and inference cost O(d*K), comparable to linear. If it does not beat label-linear (0.8524), it means interactions are not predictive or rank is too small.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.003]
L2 = 1e-4
BATCH = 64
EPOCHS = 20
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_fm(X, y, K, lr, l2, epochs, batch, rng):
    n, d = X.shape
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)
    m_linear = np.zeros_like(linear); v_linear = np.zeros_like(linear)
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_bias = 0.0; v_bias = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            x = X[idx]
            yy = y[idx]
            bsize = len(idx)
            # forward
            lin_part = np.zeros(bsize, np.float32)
            for j in range(d):
                lin_part += linear[j, x[:, j]]
            lin_part += bias
            sum_e = np.zeros((bsize, K), np.float32)
            for j in range(d):
                sum_e += emb[j, x[:, j], :]
            sq_sum_e = np.sum(sum_e * sum_e, axis=1)
            sq_emb = np.zeros(bsize, np.float32)
            for j in range(d):
                sq_emb += np.sum(emb[j, x[:, j], :] * emb[j, x[:, j], :], axis=1)
            inter = 0.5 * (sq_sum_e - sq_emb)
            score = lin_part + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad = p - yy  # (bsize,)
            # backward
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], x[:, j], grad)
            grad_bias = np.sum(grad)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                x_j = x[:, j]
                diff = sum_e - emb[j, x_j, :]  # (bsize, K)
                np.add.at(grad_emb[j], x_j, grad[:, None] * diff)
            # Adam update
            t += 1
            m_linear = beta1 * m_linear + (1 - beta1) * (grad_linear + l2 * linear)
            v_linear = beta2 * v_linear + (1 - beta2) * (grad_linear * grad_linear)
            linear -= lr * (m_linear / (1 - beta1**t)) / (np.sqrt(v_linear / (1 - beta2**t)) + eps)
            m_emb = beta1 * m_emb + (1 - beta1) * (grad_emb + l2 * emb)
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            emb -= lr * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias * grad_bias)
            bias -= lr * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)
    return linear, emb, bias

def evaluate(X, y, linear, emb, bias, K):
    n, d = X.shape
    lin_part = np.zeros(n, np.float32)
    for j in range(d):
        lin_part += linear[j, X[:, j]]
    lin_part += bias
    sum_e = np.zeros((n, K), np.float32)
    for j in range(d):
        sum_e += emb[j, X[:, j], :]
    sq_sum_e = np.sum(sum_e * sum_e, axis=1)
    sq_emb = np.zeros(n, np.float32)
    for j in range(d):
        sq_emb += np.sum(emb[j, X[:, j], :] * emb[j, X[:, j], :], axis=1)
    inter = 0.5 * (sq_sum_e - sq_emb)
    score = lin_part + inter
    pred = (score > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    best_acc = -1.0
    best_K = K_CHOICES[0]
    best_lr = LR_CHOICES[0]
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            rng = np.random.default_rng(SEED * 1000 + K * 100 + int(lr * 1000))
            linear, emb, bias = train_fm(train_X, train_y, K, lr, L2, EPOCHS, BATCH, rng)
            acc = evaluate(tune_X, tune_y, linear, emb, bias, K)
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_lr = lr
    # retrain on all data with best hyperparameters
    rng = np.random.default_rng(SEED)
    linear, emb, bias = train_fm(X, y, best_K, best_lr, L2, EPOCHS, BATCH, rng)
    # write weights.bin
    out = struct.pack("<4i", 0x464D0001, d, best_K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype(np.float32).tobytes()
    out += emb.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || d > 100 || K <= 0 || K > 32 || C != 2) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *p = (float *)(wb + 16);
    float bias = p[0];
    float *linear = p + 1;
    float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = bias;
        for (int j = 0; j < d; j++) {
            score += linear[(long)j * 256 + xi[j]];
        }
        float sum_e[32];
        for (int kk = 0; kk < K; kk++) sum_e[kk] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) sum_e[kk] += e[kk];
        }
        float sq_sum_e = 0.0f;
        for (int kk = 0; kk < K; kk++) sq_sum_e += sum_e[kk] * sum_e[kk];
        float sq_emb = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) sq_emb += e[kk] * e[kk];
        }
        score += 0.5f * (sq_sum_e - sq_emb);
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```