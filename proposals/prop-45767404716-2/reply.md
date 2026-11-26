```hypothesis
Family: label-emb-mlp. Shared byte embedding (256×D) across 16 context positions, concatenated, then one ReLU hidden layer (size H), then linear to 65 classes. Hyperparameters D,H chosen on tuning split (last 20% of train). Captures position‑independent byte similarities and pairwise interactions, improving over label‑linear’s one‑hot linear model. Failure if dev accuracy ≤ 0.3375 or cost per item ≫ label‑linear.
```

```python
"""label-emb-mlp: shared embedding + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Grid searched on tuning split
D_OPTIONS = [8, 16, 32]
H_OPTIONS = [32, 64, 128]
GRID_EPOCHS = 30  # reduced epochs for grid search


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_mlp(X, y, D, H, C, epochs, batch, lr, l2, rng):
    n, d = X.shape
    assert d == 16
    # Parameters
    E = (rng.normal(0, 0.01, (256, D))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (16 * D, H)) * math.sqrt(6.0 / (16 * D + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (batch, 16) uint8
        emb = E[Xb]                     # (batch, 16, D)
        flat = emb.reshape(len(Xb), -1) # (batch, 16*D)
        h_pre = flat @ W1 + b1
        h = np.maximum(h_pre, 0, dtype=np.float32)
        logits = h @ W2 + b2
        return flat, h_pre, h, logits

    def softmax_cross_entropy(logits, yb):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()
        probs[np.arange(len(yb)), yb] -= 1.0
        return loss, probs.astype(np.float32) / len(yb)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            flat, h_pre, h, logits = forward(Xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)

            # Gradients
            dW2 = h.T @ dlogits + l2 * W2
            db2 = dlogits.sum(0)
            dh = dlogits @ W2.T
            dh[h_pre <= 0] = 0
            dW1 = flat.T @ dh + l2 * W1
            db1 = dh.sum(0)
            dflat = dh @ W1.T                     # (batch, 16*D)
            demb = dflat.reshape(len(Xb), 16, D)  # (batch, 16, D)
            # Embedding gradient via scatter-add
            dE = np.zeros_like(E)
            np.add.at(dE, Xb.ravel(), demb.reshape(-1, D))
            dE += l2 * E

            grads = [dE, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    E, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        Xb = X[s:s+1024]
        emb = E[Xb]
        flat = emb.reshape(len(Xb), -1)
        h = np.maximum(flat @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == y[s:s+1024]).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    val_size = n // 5
    tr_X, tr_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    base_rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_D = best_H = None

    for D in D_OPTIONS:
        for H in H_OPTIONS:
            rng = np.random.default_rng(base_rng.integers(2**32))
            params = train_mlp(tr_X, tr_y, D, H, C, GRID_EPOCHS, BATCH, LR, L2, rng)
            acc = evaluate(val_X, val_y, params)
            if acc > best_acc:
                best_acc = acc
                best_D, best_H = D, H
                best_params = params
            print(f"D={D} H={H} val_acc={acc:.4f}", file=sys.stderr)

    print(f"Best: D={best_D} H={best_H} val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data with best hyperparameters
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(base_rng.integers(2**32))
    final_params = train_mlp(X, y, best_D, best_H, C, final_epochs, BATCH, LR, L2, rng)

    # Save weights
    E, W1, b1, W2, b2 = final_params
    magic = 0x454D424D  # 'EMBM' little-endian
    out = struct.pack("<4i", magic, best_D, best_H, C)
    out += E.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int D = (int)hd[1], H = (int)hd[2], C = (int)hd[3];
    if (C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * D + 16L * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W1 = E + 256L * D;
    const float *b1 = W1 + 16L * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * 16 * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < 16; j++) {
            const float *e = E + (long)xi[j] * D;
            memcpy(emb_concat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(emb_concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h; // column-major? W1 is (16*D, H) row-major: W1[row*H + h]
            for (int idx = 0; idx < 16 * D; idx++) {
                sum += emb_concat[idx] * w[idx * H];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```