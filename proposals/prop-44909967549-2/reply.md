```hypothesis
Family: label-embed (shared character embeddings + positional embeddings + 1-hidden-layer ReLU MLP).
Why: The task predicts next character from 16-byte context. Shared embeddings capture character similarities (e.g., 'a' and 'b' close) across all positions, while positional embeddings retain order information. Summing them yields a compact context vector fed to a non-linear classifier. This uses far fewer parameters than label-linear (≈15k vs 266k) and adds non-linearity, which should improve accuracy on this structured text task. If dev accuracy does not exceed label-linear's 0.3375 or CPU cost per item exceeds label-linear significantly, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"E": 16, "H": 32, "lr": 3e-3, "epochs": 50, "batch": 64},
    {"E": 32, "H": 64, "lr": 3e-3, "epochs": 50, "batch": 64},
    {"E": 32, "H": 128, "lr": 3e-3, "epochs": 50, "batch": 64},
    {"E": 64, "H": 128, "lr": 3e-3, "epochs": 50, "batch": 64},
]
SEED = 1
L2 = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_train, y_train, X_val, y_val, E, H, lr, epochs, batch, seed):
    n_train, d = X_train.shape
    C = 65
    rng = np.random.default_rng(seed)
    # Xavier initialization
    char_emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_emb = (rng.uniform(-1, 1, (d, E)) * math.sqrt(6.0 / d)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [char_emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, batch):
            idx = order[s:s + batch]
            xb = X_train[idx]
            yb = y_train[idx]
            # Forward: h = sum_j (char_emb[xb[:,j]] + pos_emb[j])
            h = np.zeros((len(idx), E), np.float32)
            for j in range(d):
                h += char_emb[xb[:, j]] + pos_emb[j]
            a = h @ W1 + b1
            hidden = np.maximum(a, 0)
            logits = hidden @ W2 + b2
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[a <= 0] = 0
            grad_W1 = h.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_h = grad_hidden @ W1.T
            # Embedding gradients
            grad_char_emb = np.zeros_like(char_emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                np.add.at(grad_char_emb, xb[:, j], grad_h)
                grad_pos_emb[j] += grad_h.sum(axis=0)
            grads = [grad_char_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    if X_val is not None:
        h_val = np.zeros((X_val.shape[0], E), np.float32)
        for j in range(d):
            h_val += char_emb[X_val[:, j]] + pos_emb[j]
        hidden_val = np.maximum(h_val @ W1 + b1, 0)
        logits_val = hidden_val @ W2 + b2
        preds = logits_val.argmax(axis=1)
        acc = (preds == y_val).mean()
        return acc, params
    return None, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    best_acc = -1.0
    best_params = None
    best_hp = None
    for hp in GRID:
        acc, params = train_eval(X_train, y_train, X_val, y_val, hp["E"], hp["H"], hp["lr"], hp["epochs"], hp["batch"], SEED)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_hp = hp
    print(f"Best hyperparams: {best_hp}, val acc: {best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    _, final_params = train_eval(X, y, None, None, best_hp["E"], best_hp["H"], best_hp["lr"], best_hp["epochs"], best_hp["batch"], SEED)
    char_emb, pos_emb, W1, b1, W2, b2 = final_params
    E, H = best_hp["E"], best_hp["H"]
    C = card["classes"]
    magic = 0x44424D45  # 'EMBD' little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += char_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *char_emb = (const float *)(hd + 5);
    const float *pos_emb = char_emb + 256L * E;
    const float *W1 = pos_emb + (long)d * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *ce = char_emb + (long)xi[j] * E;
            const float *pe = pos_emb + (long)j * E;
            for (int e = 0; e < E; e++) h[e] += ce[e] + pe[e];
        }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e = 0; e < E; e++) sum += h[e] * W1[(long)e * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```