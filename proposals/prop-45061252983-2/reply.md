```hypothesis
Family: label-embedding (position-specific embeddings + linear classifier).
Why: The linear model uses 4096 one-hot features (16 positions × 256 values) which is high-dimensional and sparse. Learned embeddings (e.g., 16×8=128 dims) capture byte similarities and position-specific patterns, improving generalization with far fewer parameters. The model remains a simple linear layer on top of embeddings, so inference stays fast (similar to label-linear). We tune embedding dimension D on the last 20% of train.in, then train on all data.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or instructions per item exceed label-linear by >20%.
```

```python
import json, struct, sys, math
import numpy as np

def read_records(path):
    data = open(path, "rb").read()
    records, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        records.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return records

def main():
    card = json.load(open(sys.argv[1]))
    X_rec = read_records(sys.argv[2])
    y_rec = read_records(sys.argv[3])
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(X_rec), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_rec], dtype=np.int64)
    n = X.shape[0]
    split = int(n * 0.8)                # last 20% for tuning
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    D_candidates = [8, 16, 32]
    best_D, best_acc = None, -1.0
    base_seed = 12345

    for D in D_candidates:
        rng = np.random.default_rng(base_seed + D)
        emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
        W   = rng.normal(0, 0.01, size=(d * D, C)).astype(np.float32)
        b   = np.zeros(C, dtype=np.float32)

        m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
        m_W,   v_W   = np.zeros_like(W),   np.zeros_like(W)
        m_b,   v_b   = np.zeros_like(b),   np.zeros_like(b)

        lr, beta1, beta2, eps = 0.01, 0.9, 0.999, 1e-8
        L2 = 1e-4
        batch_size = 64
        epochs = 20
        t = 0

        for _ in range(epochs):
            order = rng.permutation(X_tr.shape[0])
            for s in range(0, X_tr.shape[0], batch_size):
                idx = order[s:s + batch_size]
                xb, yb = X_tr[idx], y_tr[idx]
                batch = xb.shape[0]

                hidden = np.zeros((batch, d * D), dtype=np.float32)
                for j in range(d):
                    hidden[:, j * D:(j + 1) * D] = emb[j, xb[:, j], :]

                logits = hidden @ W + b
                max_logits = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(batch), yb] -= 1.0
                probs /= batch

                grad_hidden = probs @ W.T
                grad_W = hidden.T @ probs + L2 * W
                grad_b = probs.sum(axis=0)

                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], grad_hidden[:, j * D:(j + 1) * D])
                grad_emb += L2 * emb

                t += 1
                # Adam updates
                m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
                v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
                m_hat = m_emb / (1 - beta1 ** t)
                v_hat = v_emb / (1 - beta2 ** t)
                emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
                m_hat = m_W / (1 - beta1 ** t)
                v_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_hat / (np.sqrt(v_hat) + eps)

                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
                m_hat = m_b / (1 - beta1 ** t)
                v_hat = v_b / (1 - beta2 ** t)
                b -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # evaluate on tuning split
        hidden = np.zeros((X_tune.shape[0], d * D), dtype=np.float32)
        for j in range(d):
            hidden[:, j * D:(j + 1) * D] = emb[j, X_tune[:, j], :]
        logits = hidden @ W + b
        preds = logits.argmax(axis=1)
        acc = (preds == y_tune).mean()
        print(f"D={D}, tuning acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_D = acc, D

    # retrain on full data with best_D
    print(f"Best D={best_D}, retraining on full data...")
    rng = np.random.default_rng(base_seed + best_D)
    emb = rng.normal(0, 0.01, size=(d, 256, best_D)).astype(np.float32)
    W   = rng.normal(0, 0.01, size=(d * best_D, C)).astype(np.float32)
    b   = np.zeros(C, dtype=np.float32)

    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W,   v_W   = np.zeros_like(W),   np.zeros_like(W)
    m_b,   v_b   = np.zeros_like(b),   np.zeros_like(b)

    t = 0
    epochs = 30
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb, yb = X[idx], y[idx]
            batch = xb.shape[0]

            hidden = np.zeros((batch, d * best_D), dtype=np.float32)
            for j in range(d):
                hidden[:, j * best_D:(j + 1) * best_D] = emb[j, xb[:, j], :]

            logits = hidden @ W + b
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch), yb] -= 1.0
            probs /= batch

            grad_hidden = probs @ W.T
            grad_W = hidden.T @ probs + L2 * W
            grad_b = probs.sum(axis=0)

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_hidden[:, j * best_D:(j + 1) * best_D])
            grad_emb += L2 * emb

            t += 1
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
            m_hat = m_b / (1 - beta1 ** t)
            v_hat = v_b / (1 - beta2 ** t)
            b -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    magic = 0x44424D45  # "EMBD" little-endian
    out = struct.pack("<4I", magic, d, best_D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    if (hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + (long)d * 256 * D * 4 + (long)d * D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W   = emb + (long)d * 256 * D;
    const float *b   = W + (long)d * D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d * D; j++) hidden[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int dd = 0; dd < D; dd++) hidden[j * D + dd] = e[dd];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int h = 0; h < d * D; h++) sum += hidden[h] * W[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```