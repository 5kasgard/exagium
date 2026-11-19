```hypothesis
Family: label-1hot (per-pixel one-hot linear). Each pixel value (0–255) gets a class weight vector; logits are the sum of 64 vectors plus bias. This captures arbitrary per-pixel non-linearities while keeping inference identical to label-linear (64×C adds). On optdigits, pixels have only 17 used values, so the model can learn optimal non-linear responses. If dev accuracy does not exceed label-linear (0.9613), the extra capacity does not help.
```

```python
import json, math, struct, sys
import numpy as np

LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 30
MAX_EPOCHS_FULL = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_eval(lr, l2, epochs, X_train, y_train, X_val, y_val, C, d, rng):
    n_train = X_train.shape[0]
    T = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    m_T, v_T = np.zeros_like(T), np.zeros_like(T)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            Xb = X_train[idx]
            yb = y_train[idx]
            B = len(yb)
            logits = np.zeros((B, C), np.float32)
            for j in range(d):
                logits += T[j, Xb[:, j], :]
            logits += b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B
            grad_T = np.zeros_like(T)
            for j in range(d):
                np.add.at(grad_T[j], Xb[:, j], dlogits)
            grad_b = dlogits.sum(axis=0)
            grad_T += l2 * T
            grad_b += l2 * b
            t += 1
            m_T = beta1 * m_T + (1 - beta1) * grad_T
            v_T = beta2 * v_T + (1 - beta2) * grad_T * grad_T
            m_hat_T = m_T / (1 - beta1**t)
            v_hat_T = v_T / (1 - beta2**t)
            T -= lr * m_hat_T / (np.sqrt(v_hat_T) + eps)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1**t)
            v_hat_b = v_b / (1 - beta2**t)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
    n_val = X_val.shape[0]
    correct = 0
    for s in range(0, n_val, BATCH):
        Xb = X_val[s:s+BATCH]
        yb = y_val[s:s+BATCH]
        logits = np.zeros((len(yb), C), np.float32)
        for j in range(d):
            logits += T[j, Xb[:, j], :]
        logits += b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n_val, T, b

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    n_val = n // 5
    n_train = n - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    for lr in LR_GRID:
        for l2 in L2_GRID:
            acc, _, _ = train_eval(lr, l2, MAX_EPOCHS_TUNE, X_train, y_train, X_val, y_val, C, d, rng)
            print(f"lr={lr}, l2={l2}, val_acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_params = (lr, l2)
    print(f"Best: lr={best_params[0]}, l2={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)
    lr, l2 = best_params
    rng = np.random.default_rng(SEED)
    T = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    m_T, v_T = np.zeros_like(T), np.zeros_like(T)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for epoch in range(MAX_EPOCHS_FULL):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            B = len(yb)
            logits = np.zeros((B, C), np.float32)
            for j in range(d):
                logits += T[j, Xb[:, j], :]
            logits += b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B
            grad_T = np.zeros_like(T)
            for j in range(d):
                np.add.at(grad_T[j], Xb[:, j], dlogits)
            grad_b = dlogits.sum(axis=0)
            grad_T += l2 * T
            grad_b += l2 * b
            t += 1
            m_T = beta1 * m_T + (1 - beta1) * grad_T
            v_T = beta2 * v_T + (1 - beta2) * grad_T * grad_T
            m_hat_T = m_T / (1 - beta1**t)
            v_hat_T = v_T / (1 - beta2**t)
            T -= lr * m_hat_T / (np.sqrt(v_hat_T) + eps)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1**t)
            v_hat_b = v_b / (1 - beta2**t)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
    magic = 0x54424C31
    H = 0
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```