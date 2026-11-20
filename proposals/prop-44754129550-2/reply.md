```hypothesis
Family: label-embed (embedding sum + optional hidden layer). Each of the 16 positions gets a learned embedding vector (dim D). Summed embeddings are fed to a linear layer (H=0) or a one-hidden-layer ReLU MLP (H>0). This reduces parameters vs. one-hot linear (16*256*D vs 16*256*C) and captures positional interactions via the sum. Tuning D,H on the last 20% of train should beat label-linear (0.3375) or match it faster. Failure: if best dev accuracy ≤ 0.3375 or inference cost ≥ label-linear.
```
```python
"""label-embed: embedding sum + optional hidden layer for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
D_CHOICES = [16, 32, 64]
H_CHOICES = [0, 32, 64]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4244  # 'EMBD' little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, D, H, rng, epochs):
    n, d = X.shape
    # Embeddings: d positions, 256 values each, D dims
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / D)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            # Sum embeddings: (batch, D)
            h = np.zeros((len(xb), D), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            if H:
                a = h @ W1 + b1
                h_relu = np.maximum(a, 0)
                logits = h_relu @ W2 + b2
            else:
                logits = h @ W + b
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backward
            if H:
                grad_W2 = h_relu.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_h_relu = grad_logits @ W2.T
                grad_a = grad_h_relu * (a > 0)
                grad_W1 = h.T @ grad_a + L2 * W1
                grad_b1 = grad_a.sum(0)
                grad_h = grad_a @ W1.T
            else:
                grad_W = h.T @ grad_logits + L2 * W
                grad_b = grad_logits.sum(0)
                grad_h = grad_logits @ W.T
            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h)
            grad_emb += L2 * emb
            # Adam update
            t += 1
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2] if H else [grad_emb, grad_W, grad_b]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def eval_model(X, y, params, H):
    n, d = X.shape
    emb = params[0]
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W, b = params[1], params[2]
    h = np.zeros((n, emb.shape[2]), np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    if H:
        a = h @ W1 + b1
        h_relu = np.maximum(a, 0)
        logits = h_relu @ W2 + b2
    else:
        logits = h @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X)
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_D, best_H, best_params = -1, None, None, None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    for D in D_CHOICES:
        for H in H_CHOICES:
            params = train_model(X_train, y_train, C, D, H, rng, epochs)
            acc = eval_model(X_val, y_val, params, H)
            if acc > best_acc:
                best_acc, best_D, best_H, best_params = acc, D, H, params
    # Retrain on full data
    rng = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, C, best_D, best_H, rng, epochs_full)
    # Write weights
    emb = final_params[0]  # (d, 256, D)
    out = struct.pack("<5i", MAGIC, d, best_D, best_H, C)
    out += emb.astype("<f4").tobytes()
    if best_H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        out += W.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (H ? (long)D * H + H + (long)H * C + C : (long)D * C + C));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    const float *W = W1;  // reuse when H=0
    const float *b = b1;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum = malloc(sizeof(float) * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) sum[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int k = 0; k < D; k++) sum[k] += e[k];
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float v = b1[h];
                for (int k = 0; k < D; k++) v += sum[k] * W1[(long)k * H + h];
                hidden[h] = v > 0.0f ? v : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float v = b2[c];
                for (int h = 0; h < H; h++) v += hidden[h] * W2[(long)h * C + c];
                logits[c] = v;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float v = b[c];
                for (int k = 0; k < D; k++) v += sum[k] * W[(long)k * C + c];
                logits[c] = v;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```