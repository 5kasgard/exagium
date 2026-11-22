```hypothesis
Family: label-embedding-mlp. Learns a position-specific embedding for each of the 16 input bytes, sums them into a fixed-size vector, then applies an optional ReLU hidden layer before the output. This compresses the 4096-d one-hot space into a dense embedding (e.g., 32-d), freeing capacity for a hidden layer that captures cross-position interactions. Expect to beat label-linear (0.3375) on dev accuracy because embeddings share statistical strength across characters and positions, and the hidden layer adds nonlinearity. Failure: dev accuracy ≤ 0.3375 or inference cost (Ir/item) exceeds label-linear by >20%.
```

```python
import json, math, struct, sys

import numpy as np

# Tuning grid: (embedding_dim, hidden_dim). hidden_dim=0 means linear on embeddings.
GRID = [(16, 0), (16, 32), (16, 64), (32, 0), (32, 32), (32, 64), (64, 0), (64, 32), (64, 64)]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, rng):
    n = X.shape[0]
    # Embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)
            # Forward: sum embeddings
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]
            if H:
                hidden = emb_sum @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2
            else:
                logits = emb_sum @ W + b
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)
            # Backward
            if H:
                grad_W2 = hidden_relu.T @ grad_logits
                grad_b2 = grad_logits.sum(0)
                grad_hidden_relu = grad_logits @ W2.T
                grad_hidden = grad_hidden_relu * (hidden > 0)
                grad_W1 = emb_sum.T @ grad_hidden
                grad_b1 = grad_hidden.sum(0)
                grad_emb_sum = grad_hidden @ W1.T
            else:
                grad_W = emb_sum.T @ grad_logits
                grad_b = grad_logits.sum(0)
                grad_emb_sum = grad_logits @ W.T
            # L2 regularization
            if H:
                grad_W2 += L2 * W2
                grad_W1 += L2 * W1
            else:
                grad_W += L2 * W
            # Gradient for embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            # Adam update
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2] if H else [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(params, X, y, d, C, H):
    n = X.shape[0]
    emb = params[0]
    correct = 0
    for i in range(n):
        emb_sum = np.zeros(emb.shape[2], np.float32)
        for j in range(d):
            emb_sum += emb[j, X[i, j]]
        if H:
            W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
            hidden = emb_sum @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2
        else:
            W, b = params[1], params[2]
            logits = emb_sum @ W + b
        if np.argmax(logits) == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_EH = -1, None, None
    for E, H in GRID:
        params = train_model(X_tr, y_tr, d, C, E, H, rng)
        acc = evaluate(params, X_tu, y_tu, d, C, H)
        if acc > best_acc:
            best_acc, best_params, best_EH = acc, params, (E, H)
    # Retrain on full data
    E, H = best_EH
    rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, d, C, E, H, rng)
    emb = final_params[0]
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * E * 4;
    if (H) {
        expected += (long)E * H * 4;
        expected += (long)H * 4;
        expected += (long)H * C * 4;
        expected += (long)C * 4;
    } else {
        expected += (long)E * C * 4;
        expected += (long)C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wbuf + 20);
    const float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL, *W = NULL, *b = NULL;
    if (H) {
        W1 = emb + (long)d * 256 * E;
        b1 = W1 + (long)E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W = emb + (long)d * 256 * E;
        b = W + (long)E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W[(long)e * C + c];
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