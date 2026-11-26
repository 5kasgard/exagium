```hypothesis
Family: label-emb (embedding + optional hidden layer). Each of the 16 context bytes gets a learned embedding vector; concatenated embeddings feed a linear or one-hidden-layer ReLU classifier. This factorizes the 4096 one-hot features into 16×D parameters, capturing character similarities and reducing overfitting. On the tuning split we select D∈{8,16}, H∈{0,32}. Expected dev accuracy >0.34 (beating label-linear) with similar or lower CPU cost. If dev accuracy ≤0.335 or inference cost exceeds label-linear by >20%, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"D": 8, "H": 0},
    {"D": 8, "H": 32},
    {"D": 16, "H": 0},
    {"D": 16, "H": 32},
]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30
MAGIC = 0x454D4244  # "EMBD" in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, D, H, rng):
    emb = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    if H > 0:
        W1 = rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = emb + [W1, b1, W2, b2]
    else:
        W_out = rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = emb + [W_out, b_out]
    return params


def forward(params, X, d, D, H, C):
    emb = params[:d]
    emb_concat = np.concatenate([emb[j][X[:, j]] for j in range(d)], axis=1)
    if H > 0:
        W1, b1, W2, b2 = params[d:]
        h = np.maximum(emb_concat @ W1 + b1, 0)
        logits = h @ W2 + b2
        return logits, (emb_concat, h)
    else:
        W_out, b_out = params[d:]
        logits = emb_concat @ W_out + b_out
        return logits, (emb_concat,)


def loss_grad(logits, y):
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def backward(params, grad_logits, cache, X, d, D, H, C, L2):
    emb = params[:d]
    if H > 0:
        W1, b1, W2, b2 = params[d:]
        emb_concat, h = cache
        grad_W2 = h.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(0)
        grad_h = grad_logits @ W2.T
        grad_h[h <= 0] = 0
        grad_W1 = emb_concat.T @ grad_h + L2 * W1
        grad_b1 = grad_h.sum(0)
        grad_emb_concat = grad_h @ W1.T
        grad_emb = []
        for j in range(d):
            start = j * D
            end = (j + 1) * D
            g = grad_emb_concat[:, start:end]
            g_emb = np.zeros((256, D), np.float32)
            np.add.at(g_emb, X[:, j], g)
            grad_emb.append(g_emb + L2 * emb[j])
        return grad_emb + [grad_W1, grad_b1, grad_W2, grad_b2]
    else:
        W_out, b_out = params[d:]
        emb_concat, = cache
        grad_W_out = emb_concat.T @ grad_logits + L2 * W_out
        grad_b_out = grad_logits.sum(0)
        grad_emb_concat = grad_logits @ W_out.T
        grad_emb = []
        for j in range(d):
            start = j * D
            end = (j + 1) * D
            g = grad_emb_concat[:, start:end]
            g_emb = np.zeros((256, D), np.float32)
            np.add.at(g_emb, X[:, j], g)
            grad_emb.append(g_emb + L2 * emb[j])
        return grad_emb + [grad_W_out, grad_b_out]


def train(X, y, d, C, D, H, epochs, batch, lr, l2, seed):
    rng = np.random.default_rng(seed)
    params = init_params(d, C, D, H, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            logits, cache = forward(params, xb, d, D, H, C)
            grad_logits = loss_grad(logits, yb)
            grads = backward(params, grad_logits, cache, xb, d, D, H, C, l2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def accuracy(params, X, y, d, D, H, C):
    logits, _ = forward(params, X, d, D, H, C)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def save_weights(path, params, d, D, H, C):
    emb = params[:d]
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", MAGIC, d, D, H, C))
        for j in range(d):
            f.write(emb[j].astype("<f4").tobytes())
        if H > 0:
            W1, b1, W2, b2 = params[d:]
            f.write(W1.astype("<f4").tobytes())
            f.write(b1.astype("<f4").tobytes())
            f.write(W2.astype("<f4").tobytes())
            f.write(b2.astype("<f4").tobytes())
        else:
            W_out, b_out = params[d:]
            f.write(W_out.astype("<f4").tobytes())
            f.write(b_out.astype("<f4").tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    best_acc = -1.0
    best_cfg = None
    best_params = None
    for cfg in GRID:
        D, H = cfg["D"], cfg["H"]
        params = train(X_tr, y_tr, d, C, D, H, EPOCHS, BATCH, LR, L2, SEED)
        acc = accuracy(params, X_val, y_val, d, D, H, C)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params

    # Retrain on full data with best config
    D, H = best_cfg["D"], best_cfg["H"]
    final_params = train(X_all, y_all, d, C, D, H, EPOCHS, BATCH, LR, L2, SEED)
    save_weights(sys.argv[4], final_params, d, D, H, C)


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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || D < 1 || H < 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * D * 4;
    if (H > 0) {
        expected += (long)d * D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    } else {
        expected += (long)d * D * C * 4 + C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (H > 0 ? (long)d * D * H : 0);
    const float *W2 = b1 + (H > 0 ? H : 0);
    const float *b2 = W2 + (H > 0 ? (long)H * C : 0);
    const float *W_out = W1;
    const float *b_out = b1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc((long)d * D * sizeof(float));
    float *hidden = H > 0 ? malloc((long)H * sizeof(float)) : NULL;
    float *logits = malloc((long)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(emb_concat + (long)j * D, src, D * sizeof(float));
        }
        if (H > 0) {
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < d * D; v++) sum += emb_concat[v] * W1[(long)v * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int v = 0; v < d * D; v++) sum += emb_concat[v] * W_out[(long)v * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```