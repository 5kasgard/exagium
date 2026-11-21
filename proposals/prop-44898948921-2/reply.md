```hypothesis
Family: label-hashng (hashed n-gram linear). Unigrams (position,byte) and bigrams (position,byte,byte) hashed into a fixed-size table (power-of-two, 4k–32k rows) with one C-dim vector per bucket. Trained with Adam + L2. Why it should beat label-linear: captures local character dependencies (bigrams) crucial for text, while feature hashing keeps inference fast (31 table lookups + adds per item) and memory modest (~4 MB). If dev accuracy ≤ 0.34 or Ir/item ≥ 8000, the hypothesis fails.
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
CANDIDATE_H = [4096, 8192, 16384, 32768]
MULT = np.uint64(11400714819323198485)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_keys(keys, hbits):
    return ((keys * MULT) >> np.uint64(64 - hbits)).astype(np.uint32)


def train_split(X, y, H, hbits, C, n_epochs, rng):
    n, d = X.shape
    table = rng.uniform(-0.01, 0.01, (H, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    m_table = np.zeros_like(table); v_table = np.zeros_like(table)
    m_bias = np.zeros_like(bias); v_bias = np.zeros_like(bias)
    t = 0

    uni_offsets = np.arange(d, dtype=np.uint64) * 256
    bi_offsets = np.arange(d - 1, dtype=np.uint64) * 256 * 256

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            b = len(yb)

            uni_keys = xb.astype(np.uint64) + uni_offsets[None, :]
            uni_hashes = hash_keys(uni_keys, hbits)

            bi_keys = (xb[:, :-1].astype(np.uint64) * 256 + xb[:, 1:]) + bi_offsets[None, :]
            bi_hashes = hash_keys(bi_keys, hbits)

            logits = np.tile(bias, (b, 1))
            for j in range(d):
                logits += table[uni_hashes[:, j]]
            for j in range(d - 1):
                logits += table[bi_hashes[:, j]]

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(b), yb] -= 1.0
            grad = probs / b

            g_bias = grad.sum(axis=0) + L2 * bias
            g_table = np.zeros_like(table)
            for i in range(b):
                for j in range(d):
                    g_table[uni_hashes[i, j]] += grad[i]
                for j in range(d - 1):
                    g_table[bi_hashes[i, j]] += grad[i]
            g_table += L2 * table

            t += 1
            beta1, beta2 = 0.9, 0.999
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            m_table = beta1 * m_table + (1 - beta1) * g_table
            v_table = beta2 * v_table + (1 - beta2) * g_table * g_table
            m_hat = m_table / (1 - beta1 ** t)
            v_hat = v_table / (1 - beta2 ** t)
            table -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return table, bias


def evaluate(X, y, table, bias, hbits, C):
    n, d = X.shape
    uni_offsets = np.arange(d, dtype=np.uint64) * 256
    bi_offsets = np.arange(d - 1, dtype=np.uint64) * 256 * 256

    uni_keys = X.astype(np.uint64) + uni_offsets[None, :]
    uni_hashes = hash_keys(uni_keys, hbits)
    bi_keys = (X[:, :-1].astype(np.uint64) * 256 + X[:, 1:]) + bi_offsets[None, :]
    bi_hashes = hash_keys(bi_keys, hbits)

    logits = np.tile(bias, (n, 1))
    for j in range(d):
        logits += table[uni_hashes[:, j]]
    for j in range(d - 1):
        logits += table[bi_hashes[:, j]]

    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X)

    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_H, best_acc = None, -1
    for H in CANDIDATE_H:
        hbits = H.bit_length() - 1
        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        rng = np.random.default_rng(SEED)
        table, bias = train_split(X_train, y_train, H, hbits, C, n_epochs, rng)
        acc = evaluate(X_tune, y_tune, table, bias, hbits, C)
        print(f"H={H}, tune_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    print(f"Best H={best_H}, retraining on all {n} samples", file=sys.stderr)
    hbits = best_H.bit_length() - 1
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    table, bias = train_split(X, y, best_H, hbits, C, n_epochs, rng)

    hdr = struct.pack("<6I", 0x484E4731, d, best_H, C, 2, hbits)
    out = hdr + bias.astype("<f4").tobytes() + table.astype("<f4").tobytes()
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

static inline uint32_t hash_key(uint64_t key, int hbits) {
    return (key * 11400714819323198485ULL) >> (64 - hbits);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wdata = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wdata;
    if (hd[0] != 0x484E4731) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], order = hd[4], hbits = hd[5];
    if (d != 16 || C != 65 || order != 2) { fprintf(stderr, "bad weights: params\n"); return 3; }
    if (H <= 0 || (H & (H - 1)) != 0) { fprintf(stderr, "bad weights: H not power of 2\n"); return 3; }
    if (hbits != 31 - __builtin_clz(H)) { fprintf(stderr, "bad weights: hbits mismatch\n"); return 3; }
    long expected = 24 + 4L * C + 4L * H * C;
    if (nw != expected) { fprintf(stderr, "bad weights: size %ld != %ld\n", nw, expected); return 3; }

    float *bias = (float *)(wdata + 24);
    float *table = (float *)(wdata + 24 + 4L * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(logits, bias, sizeof(float) * C);

        for (int j = 0; j < d; j++) {
            uint64_t key = (uint64_t)j * 256 + xi[j];
            uint32_t h = hash_key(key, hbits);
            float *row = table + (size_t)h * C;
            for (int c = 0; c < C; c++) logits[c] += row[c];
        }
        for (int j = 0; j < d - 1; j++) {
            uint64_t key = ((uint64_t)j * 256 + xi[j]) * 256 + xi[j + 1];
            uint32_t h = hash_key(key, hbits);
            float *row = table + (size_t)h * C;
            for (int c = 0; c < C; c++) logits[c] += row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```