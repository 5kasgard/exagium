```hypothesis
The label-linear model uses a linear classifier on one-hot encoded bytes (16×256×65 parameters). An embedding-MLP factorizes the first layer: a shared 256×E embedding plus per-position biases, then a small ReLU layer (H units) and output. This reduces parameters dramatically (≈8k vs 266k) while capturing non-linear interactions, improving generalization on the small tinyshakespeare dataset. If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2×, the family is not worthwhile.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters (chosen on tuning split: last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E = 8        # embedding dimension
H = 32       # hidden layer size


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def relu(x):
    return np.maximum(x, 0, dtype=np.float32)


def softmax_grad(logits, y):
    # logits: (B, C), y: (B,)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m, dtype=np.float32)
    p = e / e.sum(axis=1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    return p / len(y)


def evaluate(X, y, emb, pos_bias, W1, b1, W2, b2):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        # (B, 16, E)
        h_pre = b1 + np.sum((emb[Xb] + pos_bias) @ W1, axis=1)
        h = relu(h_pre)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_bias = (rng.uniform(-1, 1, (d, E)) * math.sqrt(6.0 / d)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d, E, H)) * math.sqrt(6.0 / (E * d))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, pos_bias, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    best_val = -1.0
    best_params = None
    t = 0

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            B = len(Xb)

            # Forward
            x_emb = emb[Xb] + pos_bias                 # (B, d, E)
            h_pre = b1 + np.sum(x_emb @ W1, axis=1)    # (B, H)
            h = relu(h_pre)
            logits = h @ W2 + b2

            # Backward
            g = softmax_grad(logits, yb)               # (B, C)
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h_pre > 0)                # (B, H)
            gb1 = gh.sum(0)
            # gh per position
            gW1 = np.zeros_like(W1)
            for j in range(d):
                gW1[j] = x_emb[:, j].T @ gh + L2 * W1[j]
            # grad for x_emb: (B, d, E)
            gx_emb = np.zeros_like(x_emb)
            for j in range(d):
                gx_emb[:, j] = gh @ W1[j].T
            # grad for emb and pos_bias
            gemb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(gemb, Xb[:, j], gx_emb[:, j])
            gpos_bias = gx_emb.sum(0) + L2 * pos_bias
            gemb += L2 * emb

            grads = [gemb, gpos_bias, gW1, gb1, gW2, gb2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        val_acc = evaluate(X_val, y_val, emb, pos_bias, W1, b1, W2, b2)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on all data with best hyperparameters (same epochs as best epoch? Simpler: continue from best for a few more epochs)
    # For simplicity, we just use the best params and do a few more epochs on all data.
    emb, pos_bias, W1, b1, W2, b2 = best_params
    params = [emb, pos_bias, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(5):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(Xb)

            x_emb = emb[Xb] + pos_bias
            h_pre = b1 + np.sum(x_emb @ W1, axis=1)
            h = relu(h_pre)
            logits = h @ W2 + b2

            g = softmax_grad(logits, yb)
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h_pre > 0)
            gb1 = gh.sum(0)
            gW1 = np.zeros_like(W1)
            for j in range(d):
                gW1[j] = x_emb[:, j].T @ gh + L2 * W1[j]
            gx_emb = np.zeros_like(x_emb)
            for j in range(d):
                gx_emb[:, j] = gh @ W1[j].T
            gemb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(gemb, Xb[:, j], gx_emb[:, j])
            gpos_bias = gx_emb.sum(0) + L2 * pos_bias
            gemb += L2 * emb

            grads = [gemb, gpos_bias, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic, d, E, H, C; then emb[256*E], pos_bias[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C]
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += pos_bias.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding-MLP family for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic(0x454D424D), d, E, H, C (int32);
 * float32: emb[256*E], pos_bias[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C] */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 20 +
        4L*256*E + 4L*d*E + 4L*d*E*H + 4L*H + 4L*H*C + 4L*C;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float*)(hd + 5);
    const float *pos_bias = emb + 256*E;
    const float *W1 = pos_bias + d*E;
    const float *b1 = W1 + (long)d*E*H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H*C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_pre = malloc(sizeof(float)*H);
    float *h = malloc(sizeof(float)*H);
    float *scores = malloc(sizeof(float)*C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h_pre = b1
        memcpy(h_pre, b1, sizeof(float)*H);

        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j]*E;
            const float *pb = pos_bias + j*E;
            const float *w = W1 + (long)j*E*H;
            // compute e+pb -> tmp[E], then tmp @ w[E x H] add to h_pre
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < E; v++) {
                    sum += (e[v] + pb[v]) * w[(long)v*H + u];
                }
                h_pre[u] += sum;
            }
        }

        // ReLU
        for (int u = 0; u < H; u++) h[u] = h_pre[u] > 0.0f ? h_pre[u] : 0.0f;

        // scores = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u*C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```