```hypothesis
Family: label-embed. Learns per-position byte embeddings (size E) summed into a context vector, then a linear layer to 65 classes. The low-rank factorization of the linear model's weight matrix captures character similarities and should improve accuracy over label-linear (0.3375) with similar inference cost (E=64 gives ~5k ops vs 1k for linear). If dev accuracy does not exceed 0.34, the factorization is too restrictive for this task.
```

```python
"""label-embed: per-position byte embeddings summed, then linear to classes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SEARCH_EPOCHS = 30
E_CANDIDATES = [32, 64, 128, 256]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, V, W, b, d, C):
    n = len(X)
    correct = 0
    batch = 1024
    E = V.shape[1]
    for start in range(0, n, batch):
        end = min(start + batch, n)
        xb = X[start:end]
        yb = y[start:end]
        emb_sum = np.zeros((len(xb), E), np.float32)
        for pos in range(d):
            indices = xb[:, pos] + pos * 256
            emb_sum += V[indices]
        logits = emb_sum @ W + b
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == yb)
    return correct / n


def train_model(X_train, y_train, X_val, y_val, E, epochs, seed):
    d = 16
    C = 65
    rng = np.random.default_rng(seed)
    V = rng.normal(0, 0.01, (d * 256, E)).astype(np.float32)
    W = rng.normal(0, 0.01, (E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = len(X_train)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            xb = X_train[idx]
            yb = y_train[idx]
            bsize = len(xb)

            emb_sum = np.zeros((bsize, E), np.float32)
            for pos in range(d):
                indices = xb[:, pos] + pos * 256
                emb_sum += V[indices]

            logits = emb_sum @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            grad_logits = probs
            grad_logits[np.arange(bsize), yb] -= 1.0
            grad_logits /= bsize

            grad_W = emb_sum.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)
            grad_emb = grad_logits @ W.T

            grad_V = np.zeros_like(V)
            for pos in range(d):
                indices = xb[:, pos] + pos * 256
                np.add.at(grad_V, indices, grad_emb)

            grad_V += L2 * V
            grad_W += L2 * W

            t += 1
            m_V = 0.9 * m_V + 0.1 * grad_V
            v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
            m_V_hat = m_V / (1 - 0.9 ** t)
            v_V_hat = v_V / (1 - 0.999 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + 1e-8)

            m_W = 0.9 * m_W + 0.1 * grad_W
            v_W = 0.999 * v_W + 0.001 * grad_W * grad_W
            m_W_hat = m_W / (1 - 0.9 ** t)
            v_W_hat = v_W / (1 - 0.999 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            m_b_hat = m_b / (1 - 0.9 ** t)
            v_b_hat = v_b / (1 - 0.999 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

    val_acc = evaluate(X_val, y_val, V, W, b, d, C)
    return V, W, b, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    n_val = n // 5
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    best_E = 64
    best_val_acc = -1.0
    for E in E_CANDIDATES:
        V, W, b, val_acc = train_model(X_train, y_train, X_val, y_val, E, SEARCH_EPOCHS, SEED)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_E = E

    V, W, b, _ = train_model(X, y, X_val, y_val, best_E, MAX_EPOCHS, SEED)

    magic = 0x44424D45  # 'EMBD' little-endian
    header = struct.pack("<4i", magic, d, best_E, C)
    out = header + V.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte class index. */
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
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *V = (const float *)(hd + 4);
    const float *W = V + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += v[e];
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W[(long)e * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```