```hypothesis
Family: label-embmlp (embedding + one ReLU layer). Learned per-position byte embeddings capture local statistics; a small dense ReLU layer models cross-position interactions crucial for language. Inference stays fast via fused embedding lookup and two matrix multiplies. If dev accuracy ≤ 0.34, the added non-linearity does not justify the cost over label-linear.
```

```python
"""label-embmlp: per-position byte embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, os, struct, sys

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

SEED = 1
BATCH = 64
MAX_EPOCHS_GRID = 20
MAX_EPOCHS_FINAL = 30
LR_GRID = [1e-3, 3e-3]
D_GRID = [16, 32]
H_GRID = [32, 64, 128]
MAGIC = 0x454D424D  # 'EMBM' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, D, H, C, rng):
    # embeddings: [d, 256, D]
    emb = (rng.normal(0, 1/math.sqrt(D), (d, 256, D))).astype(np.float32)
    # W1: [d*D, H]
    W1 = (rng.normal(0, math.sqrt(2.0/(d*D)), (d*D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C]
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(X, emb, W1, b1, W2, b2):
    """X: [N, d] uint8. Returns logits [N, C]."""
    N, d = X.shape
    D = emb.shape[2]
    H = W1.shape[1]
    # embed
    flat = np.empty((N, d*D), np.float32)
    for j in range(d):
        flat[:, j*D:(j+1)*D] = emb[j, X[:, j]]
    # hidden
    h = flat @ W1 + b1
    np.maximum(h, 0, out=h)
    # output
    logits = h @ W2 + b2
    return logits


def loss_grad(logits, y):
    """Cross-entropy gradient wrt logits. Returns [N, C] float32."""
    N = logits.shape[0]
    maxv = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - maxv)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N
    return probs.astype(np.float32)


def backward(grad_out, h, flat, W2, W1, emb, X, d, D, H, C, L2):
    """Compute gradients for all params. Returns tuple of grads."""
    N = grad_out.shape[0]
    # grad W2, b2
    gW2 = h.T @ grad_out + L2 * W2
    gb2 = grad_out.sum(axis=0)
    # grad h
    gh = grad_out @ W2.T
    gh[h == 0] = 0.0  # ReLU derivative
    # grad W1, b1
    gW1 = flat.T @ gh + L2 * W1
    gb1 = gh.sum(axis=0)
    # grad flat -> grad emb
    gflat = gh @ W1.T  # [N, d*D]
    # accumulate into emb grads
    gemb = np.zeros_like(emb)
    for j in range(d):
        idx = X[:, j]
        np.add.at(gemb[j], idx, gflat[:, j*D:(j+1)*D])
    return gemb, gW1, gb1, gW2, gb2


def train_epoch(X, y, emb, W1, b1, W2, b2, lr, batch, rng, L2=1e-4):
    N = X.shape[0]
    order = rng.permutation(N)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for s in range(0, N, batch):
        idx = order[s:s+batch]
        xb, yb = X[idx], y[idx]
        # forward
        flat = np.empty((len(idx), X.shape[1]*emb.shape[2]), np.float32)
        for j in range(X.shape[1]):
            flat[:, j*emb.shape[2]:(j+1)*emb.shape[2]] = emb[j, xb[:, j]]
        h = flat @ W1 + b1
        np.maximum(h, 0, out=h)
        logits = h @ W2 + b2
        # loss grad
        g = loss_grad(logits, yb)
        # backward
        gemb, gW1, gb1, gW2, gb2 = backward(g, h, flat, W2, W1, emb, xb,
                                            X.shape[1], emb.shape[2], W1.shape[1], W2.shape[1], L2)
        grads = [gemb, gW1, gb1, gW2, gb2]
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * gr
            v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def accuracy(X, y, emb, W1, b1, W2, b2):
    logits = forward(X, emb, W1, b1, W2, b2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(emb, W1, b1, W2, b2, d, D, H, C):
    """Serialize to bytes."""
    out = struct.pack("<5i", MAGIC, d, D, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # split: last 20% for tuning
    n = len(X)
    val_size = n // 5
    tr_X, val_X = X[:-val_size], X[-val_size:]
    tr_y, val_y = y[:-val_size], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for D in D_GRID:
        for H in H_GRID:
            for lr in LR_GRID:
                # fresh init
                emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
                for epoch in range(MAX_EPOCHS_GRID):
                    train_epoch(tr_X, tr_y, emb, W1, b1, W2, b2, lr, BATCH, rng)
                acc = accuracy(val_X, val_y, emb, W1, b1, W2, b2)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (D, H, lr)
                    # deep copy
                    best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    print(f"Best cfg: D={best_cfg[0]}, H={best_cfg[1]}, lr={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    D, H, lr = best_cfg
    emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    for epoch in range(MAX_EPOCHS_FINAL):
        train_epoch(X, y, emb, W1, b1, W2, b2, lr, BATCH, rng)

    # write weights
    out_bytes = pack_weights(emb, W1, b1, W2, b2, d, D, H, C)
    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20 + (long)d * 256 * D * 4 + (long)d * D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc((size_t)d * D * sizeof(float));
    float *h = malloc((size_t)H * sizeof(float));
    float *logits = malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // embed
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, src, D * sizeof(float));
        }
        // hidden = ReLU(flat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) sum += flat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        // logits = h * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```