```hypothesis
Family: label-fm (Factorization Machine with field-wise latent vectors).
Why: Adult data has strong feature interactions (e.g., education×occupation, marital×relationship) that linear models miss. FM captures all pairwise interactions via low-rank dot products (k=8) with only ~32K parameters. Inference computes 91 dot products of size 8 per item (~750 FMAs), targeting ~1500 Ir/item — faster than label-mlp16 (2325) while exceeding label-linear accuracy (0.8524).
Failure: If dev accuracy ≤ 0.8524 or Ir/item ≥ 2500, the model doesn't justify its complexity over label-linear or label-mlp16.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_EPOCHS = 5
K_CHOICES = [4, 8, 16]

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fm(X, y, k, epochs, rng, w0_init=None, w_init=None, V_init=None):
    n, d = X.shape
    if w0_init is None:
        w0 = np.float32(0.0)
    else:
        w0 = w0_init
    if w_init is None:
        w = np.zeros((d, 256), dtype=np.float32)
    else:
        w = w_init
    if V_init is None:
        V = (rng.uniform(-1, 1, (d, 256, k)) * math.sqrt(2.0 / k)).astype(np.float32)
    else:
        V = V_init

    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    steps_per_epoch = max(1, n // BATCH)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            b = len(idx)

            w_i = w[np.arange(d), Xb.T].T
            V_i = V[np.arange(d), Xb.T].transpose(1, 0, 2)

            sum_V = V_i.sum(axis=1)
            sum_V_sq = (sum_V * sum_V).sum(axis=1)
            V_i_sq = (V_i * V_i).sum(axis=(1,2))
            interaction = 0.5 * (sum_V_sq - V_i_sq)

            linear = w_i.sum(axis=1)
            score = w0 + linear + interaction

            p = sigmoid(score)
            grad_score = p - yb

            g_w0 = grad_score.mean()

            fields = np.arange(d)
            field_idx = np.tile(fields, b)
            value_idx = Xb.ravel()
            g_rep = np.repeat(grad_score, d)

            grad_w = np.zeros_like(w)
            np.add.at(grad_w, (field_idx, value_idx), g_rep)

            sum_V_rep = np.repeat(sum_V, d, axis=0)
            V_i_flat = V_i.reshape(b*d, k)
            grad_V_flat = g_rep[:, None] * (sum_V_rep - V_i_flat)
            grad_V = np.zeros_like(V)
            np.add.at(grad_V, (field_idx, value_idx), grad_V_flat)

            grad_w += L2 * w
            grad_V += L2 * V

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1**t)
            v_w0_hat = v_w0 / (1 - beta2**t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            m_w_hat = m_w / (1 - beta1**t)
            v_w_hat = v_w / (1 - beta2**t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    return w0, w, V

def evaluate(X, y, w0, w, V):
    n = X.shape[0]
    d = X.shape[1]
    batch_size = 1024
    correct = 0
    for i in range(0, n, batch_size):
        Xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        b = Xb.shape[0]
        w_i = w[np.arange(d), Xb.T].T
        V_i = V[np.arange(d), Xb.T].transpose(1, 0, 2)
        sum_V = V_i.sum(axis=1)
        interaction = 0.5 * ((sum_V * sum_V).sum(axis=1) - (V_i * V_i).sum(axis=(1,2)))
        linear = w_i.sum(axis=1)
        score = w0 + linear + interaction
        pred = (score > 0).astype(np.int64)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]

    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_k = 8
    best_acc = -1.0

    for k in K_CHOICES:
        rng = np.random.default_rng(SEED + k)
        w0, w, V = train_fm(X_train, y_train, k, TUNE_EPOCHS, rng)
        acc = evaluate(X_tune, y_tune, w0, w, V)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    rng = np.random.default_rng(SEED + best_k)
    w0, w, V = train_fm(X, y, best_k, min(MAX_EPOCHS, math.ceil(SAMPLES / n)), rng)

    magic = 0x464D0001
    header = struct.pack("<4i", magic, d, best_k, C)
    out = header + struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()

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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(wb + 16);
    float w0 = ptr[0];
    const float *w = ptr + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *V_i = (float *)malloc(sizeof(float) * d * k);
    float *sum_V = (float *)malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float linear = 0.0f;
        for (int j = 0; j < d; j++) {
            linear += w[(long)j * 256 + xi[j]];
        }

        for (int j = 0; j < d; j++) {
            const float *v_src = V + ((long)j * 256 + xi[j]) * k;
            float *v_dst = V_i + j * k;
            memcpy(v_dst, v_src, sizeof(float) * k);
        }

        memset(sum_V, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *v = V_i + j * k;
            for (int f = 0; f < k; f++) sum_V[f] += v[f];
        }

        float sum_V_sq = 0.0f;
        for (int f = 0; f < k; f++) sum_V_sq += sum_V[f] * sum_V[f];
        float V_i_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V_i + j * k;
            for (int f = 0; f < k; f++) V_i_sq += v[f] * v[f];
        }
        float interaction = 0.5f * (sum_V_sq - V_i_sq);

        float score = w0 + linear + interaction;
        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(V_i);
    free(sum_V);
    return 0;
}
```