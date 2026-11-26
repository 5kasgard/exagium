```hypothesis
label-dense-mlp: a one-hidden-layer MLP with dense weight matrices (no per-feature lookup tables), folding input standardization into the first layer. For optdigits' 64 ordinal features this eliminates 64 cache misses/item and reduces memory traffic from ~650 KB to ~4 KB. With H=8 tuned on the last 20% of train, it targets >0.965 accuracy at <4000 Ir/item (vs label-linear's 0.9613 at 5058 Ir/item). Failure: accuracy ≤0.9613 or Ir/item ≥5058.
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
CANDIDATE_H = [4, 8, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_mlp(X, y, H, C, rng, n_epochs):
    n, d = X.shape
    W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x, yy = X[idx], y[idx]
            a = x @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = (probs / len(yy)).astype(np.float32)
            dW2 = h.T @ dlogits + L2 * W2
            db2 = dlogits.sum(0)
            dh = dlogits @ W2.T
            da = dh * (a > 0)
            dW1 = x.T @ da + L2 * W1
            db1 = da.sum(0)
            grads = [dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, params):
    W1, b1, W2, b2 = params
    a = X @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    return (logits.argmax(axis=1) == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_all.shape[1]
    n = X_all.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_train_s = ((X_train - mu) / sd).astype(np.float32)
    X_tune_s = ((X_tune - mu) / sd).astype(np.float32)
    rng = np.random.default_rng(SEED)
    best_H, best_acc = 0, -1.0
    for H in CANDIDATE_H:
        params = train_mlp(X_train_s, y_train, H, C, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train))))
        acc = evaluate(X_tune_s, y_tune, params)
        if acc > best_acc:
            best_acc, best_H = acc, H
    X_full_s = ((X_all - mu) / sd).astype(np.float32)
    rng = np.random.default_rng(SEED)
    W1, b1, W2, b2 = train_mlp(X_full_s, y_all, best_H, C, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n)))
    W1_folded = (W1 / sd).astype(np.float32)
    b1_folded = (b1 - (mu / sd) @ W1).astype(np.float32)
    magic = 0x44454E53
    out = struct.pack("<4i", magic, d, best_H, C)
    out += W1_folded.astype("<f4").tobytes()
    out += b1_folded.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x44454E53) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W1 = (const float *)(hd + 4);
    const float *b1 = W1 + (long)d * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < d; j++) sum += xi[j] * W1[(long)j * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            sc[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```