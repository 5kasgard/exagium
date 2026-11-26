```hypothesis
Family: label-embed-mlp. Uses per-position byte embeddings (shared across positions) followed by a single ReLU hidden layer. This captures positional n-gram interactions more efficiently than one-hot linear models, and the low-dimensional embeddings allow a wider hidden layer at similar compute cost. Tuning embedding dimension and hidden size on the last 20% of train.in should yield a better accuracy/compute trade-off. Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost > 10k Ir/item.
```
```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters (can be tuned on the tuning split)
SEED = 12345
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Grid for tuning: (embedding_dim, hidden_size). hidden_size=0 means linear.
TUNE_GRID = [(4, 0), (4, 32), (4, 64), (8, 0), (8, 32), (8, 64), (16, 0), (16, 32), (16, 64)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_val, y_val, d, D, H, C, rng):
    n_tr, n_val = X_tr.shape[0], X_val.shape[0]
    # Initialize parameters
    # Embeddings: 16 * 256 * D
    emb = (rng.uniform(-1, 1, (16, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (16 * D, H)) * math.sqrt(6.0 / (16 * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W_out = (rng.uniform(-1, 1, (16 * D, C)) * math.sqrt(6.0 / (16 * D))).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, params):
        # X: (n, 16) uint8
        n = X.shape[0]
        emb, *rest = params
        # Embedding lookup: (n, 16, D)
        emb_seq = emb[X]  # shape (n, 16, D)
        flat = emb_seq.reshape(n, 16 * D)
        if H:
            W1, b1, W2, b2 = rest
            hidden = np.maximum(flat @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            W_out, b_out = rest
            logits = flat @ W_out + b_out
        return logits

    def loss_grad(logits, y):
        # logits: (n, C), y: (n,)
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        grad = probs / n
        return grad.astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            emb, *rest = params
            emb_seq = emb[Xb]  # (B, 16, D)
            flat = emb_seq.reshape(len(yb), 16 * D)
            if H:
                W1, b1, W2, b2 = rest
                pre = flat @ W1 + b1
                hidden = np.maximum(pre, 0)
                logits = hidden @ W2 + b2
            else:
                W_out, b_out = rest
                logits = flat @ W_out + b_out
            # Loss gradient
            g = loss_grad(logits, yb)  # (B, C)
            # Backward
            if H:
                W1, b1, W2, b2 = rest
                # grad on logits -> g
                grad_W2 = hidden.T @ g
                grad_b2 = g.sum(0)
                grad_hidden = g @ W2.T
                grad_hidden[pre <= 0] = 0
                grad_W1 = flat.T @ grad_hidden
                grad_b1 = grad_hidden.sum(0)
                grad_flat = grad_hidden @ W1.T
                # Embedding gradients
                grad_emb_seq = grad_flat.reshape(len(yb), 16, D)
                grad_emb = np.zeros_like(emb)
                np.add.at(grad_emb, Xb, grad_emb_seq)
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                W_out, b_out = rest
                grad_W_out = flat.T @ g
                grad_b_out = g.sum(0)
                grad_flat = g @ W_out.T
                grad_emb_seq = grad_flat.reshape(len(yb), 16, D)
                grad_emb = np.zeros_like(emb)
                np.add.at(grad_emb, Xb, grad_emb_seq)
                grads = [grad_emb, grad_W_out, grad_b_out]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # L2 regularization (weight decay)
                if i != 0:  # skip embeddings? apply to all for simplicity
                    p -= LR * L2 * p

    # Evaluate on validation
    logits_val = forward(X_val, params)
    pred = logits_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    # Split train into train_tune (80%) and tune (20%) in file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_acc = -1
    best_params = None
    best_D = best_H = None

    for idx, (D, H) in enumerate(TUNE_GRID):
        rng = np.random.default_rng(SEED + idx)
        acc, params = train_eval(X_tr, y_tr, X_val, y_val, d, D, H, C, rng)
        print(f"Tune D={D} H={H}: acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_D, best_H = D, H

    print(f"Best config: D={best_D} H={best_H} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full training set with best config
    rng = np.random.default_rng(SEED + 999)
    _, best_params = train_eval(X, y, X[:1], y[:1], d, best_D, best_H, C, rng)  # dummy val, we just need trained params

    # Pack weights.bin
    # Format: magic, d, D, H, C, then embeddings (16*256*D), then if H: W1, b1, W2, b2 else: W_out, b_out
    magic = 0x454D424C  # "LBME"
    emb = best_params[0]
    out = struct.pack("<5i", magic, d, best_D, best_H, C)
    out += emb.astype("<f4").tobytes()
    if best_H:
        W1, b1, W2, b2 = best_params[1:]
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W_out, b_out = best_params[1:]
        out += W_out.astype("<f4").tobytes() + b_out.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL, *W_out = NULL, *b_out = NULL;
    size_t emb_size = 16 * 256 * D;
    if (H) {
        W1 = emb + emb_size;
        b1 = W1 + 16 * D * H;
        W2 = b1 + H;
        b2 = W2 + H * C;
        long expected = 20 + 4 * (emb_size + 16 * D * H + H + H * C + C);
        if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    } else {
        W_out = emb + emb_size;
        b_out = W_out + 16 * D * C;
        long expected = 20 + 4 * (emb_size + 16 * D * C + C);
        if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * 16 * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < 16; j++) {
            const float *e = emb + (j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }

        if (H) {
            // Hidden layer: ReLU(concat * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int w = 0; w < 16 * D; w++) sum += concat[w] * W1[w * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer: hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
                logits[c] = sum;
            }
        } else {
            // Linear: concat * W_out + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int w = 0; w < 16 * D; w++) sum += concat[w] * W_out[w * C + c];
                logits[c] = sum;
            }
        }

        // Argmax
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat);
    if (hidden) free(hidden);
    free(logits);
    if (x) free(x);
    free(hd);
    return 0;
}
```