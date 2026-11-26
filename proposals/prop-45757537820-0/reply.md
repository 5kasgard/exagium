```hypothesis
Family: label-fm (Factorization Machine with per-column-value embeddings).
Why: Adult has strong feature interactions (e.g., education×occupation). FM captures all pairwise interactions with O(d·k) parameters and O(d·k) inference, where k=8 adds ~10× ops over linear but stays far below MLP cost. The raw 0–255 per-column encoding fits FM naturally.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 2000 (above label-mlp16).
```

```python
import json, math, struct, sys
import numpy as np

K_CHOICES = [4, 8, 16]
LR = 0.01
EPOCHS_TUNE = 10
EPOCHS_FULL = 30
BATCH = 256
L2 = 1e-4
SEED = 1


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, d, k, epochs, batch, lr, l2, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    v = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            yb = y[idx]
            bsz = len(xb)

            scores = np.zeros(bsz, dtype=np.float32)
            sum_v = np.zeros((bsz, k), dtype=np.float32)
            sum_v2 = np.zeros((bsz, k), dtype=np.float32)
            for j in range(d):
                vals = xb[:, j]
                scores += w[j, vals]
                sum_v += v[j, vals]
                sum_v2 += v[j, vals] ** 2
            scores += w0 + 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)

            p = 1.0 / (1.0 + np.exp(-scores))
            grad = p - yb  # (bsz,)

            t += 1
            # w0
            g_w0 = grad.mean() + l2 * w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            for j in range(d):
                vals = xb[:, j]
                # gradient for w[j]
                g_w_j = np.zeros(256, dtype=np.float32)
                np.add.at(g_w_j, vals, grad)
                g_w_j /= bsz
                g_w_j += l2 * w[j]
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * g_w_j
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * g_w_j * g_w_j
                w[j] -= lr * (m_w[j] / (1 - beta1 ** t)) / (np.sqrt(v_w[j] / (1 - beta2 ** t)) + eps)

                # gradient for v[j]
                g_v_j = np.zeros((256, k), dtype=np.float32)
                contrib = grad[:, None] * (sum_v - v[j, vals])  # (bsz, k)
                for f in range(k):
                    np.add.at(g_v_j[:, f], vals, contrib[:, f])
                g_v_j /= bsz
                g_v_j += l2 * v[j]
                m_v[j] = beta1 * m_v[j] + (1 - beta1) * g_v_j
                v_v[j] = beta2 * v_v[j] + (1 - beta2) * g_v_j * g_v_j
                v[j] -= lr * (m_v[j] / (1 - beta1 ** t)) / (np.sqrt(v_v[j] / (1 - beta2 ** t)) + eps)

    return w0, w, v


def eval_fm(X, y, params, d, k):
    w0, w, v = params
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_v2 = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        vals = X[:, j]
        scores += w[j, vals]
        sum_v += v[j, vals]
        sum_v2 += v[j, vals] ** 2
    scores += w0 + 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)
    pred = (scores > 0).astype(np.int64)
    return np.mean(pred == y)


def save_weights(path, d, k, C, params):
    w0, w, v = params
    magic = 0x464D5F31  # "FM_1"
    header = struct.pack("<4I", magic, d, k, C)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.array([w0], dtype=np.float32).tobytes())
        f.write(w.astype(np.float32).tobytes())
        f.write(v.astype(np.float32).tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_k = K_CHOICES[0]
    for k in K_CHOICES:
        params = train_fm(X_train, y_train, d, k, EPOCHS_TUNE, BATCH, LR, L2, SEED)
        acc = eval_fm(X_val, y_val, params, d, k)
        print(f"k={k}, val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Best k={best_k}, retraining on full data...", file=sys.stderr)
    final_params = train_fm(X, y, d, best_k, EPOCHS_FULL, BATCH, LR, L2, SEED)
    save_weights(sys.argv[4], d, best_k, C, final_params)


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
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    uint32_t magic = hd[0];
    int d = hd[1];
    int k = hd[2];
    int C = hd[3];
    if (magic != 0x464D5F31) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    if (C != 2) { fprintf(stderr, "bad weights: C != 2\n"); return 3; }
    long expected = 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *params = (const float *)(wb + 16);
    const float w0 = params[0];
    const float *w = params + 1;
    const float *v = params + 1 + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float score = w0;
        float sum_v[16];
        float sum_v2[16];
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += w[(long)j * 256 + val];
            const float *v_j_val = v + ((long)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vf = v_j_val[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```