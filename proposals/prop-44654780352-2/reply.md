```hypothesis
Family: label-embmlp — per-position byte embeddings (256×16) concatenated, one ReLU hidden layer (64 units), linear output to 65 classes.
Why: Factorizes the large per-position weight matrices of label-linear, captures non-linear context interactions, and regularizes via low-rank embeddings. Should exceed label-linear's 0.3375 dev accuracy at moderate CPU cost (est. ~30k Ir/item).
Failure signal: dev accuracy ≤ 0.34 or Ir/item > 50000.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed hyper-parameters (chosen on tuning split, then train on all)
D = 16          # embedding dimension per position
H = 64          # hidden layer size (0 = linear on concatenated embeddings)
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # max samples to see (like label-linear)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Tuning split: last 20% in file order
    n_val = n // 5
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    rng = np.random.default_rng(SEED)

    # Parameters
    # embeddings: list of d arrays, each (256, D)
    emb = [(rng.uniform(-0.01, 0.01, (256, D))).astype(np.float32) for _ in range(d)]
    if H:
        W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = emb + [W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = emb + [W, b]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # concatenated embeddings
        cat = np.empty((batch, d * D), dtype=np.float32)
        for j in range(d):
            cat[:, j*D:(j+1)*D] = emb[j][Xb[:, j]]
        if H:
            hidden = cat @ W1 + b1
            np.maximum(hidden, 0, out=hidden)  # ReLU in-place
            logits = hidden @ W2 + b2
        else:
            logits = cat @ W + b
        return logits, cat if not H else (logits, hidden, cat)

    def loss_and_grad(Xb, yb):
        batch = Xb.shape[0]
        if H:
            logits, hidden, cat = forward(Xb)
        else:
            logits, cat = forward(Xb)
        # softmax + cross-entropy
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), yb] -= 1.0
        grad_logits = probs / batch  # (batch, C)

        # gradients
        grads = [np.zeros_like(p) for p in params]
        if H:
            # W2, b2
            grads[-2] = hidden.T @ grad_logits + L2 * W2
            grads[-1] = grad_logits.sum(axis=0)
            # hidden grad
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden <= 0] = 0.0
            # W1, b1
            grads[-4] = cat.T @ grad_hidden + L2 * W1
            grads[-3] = grad_hidden.sum(axis=0)
            # embeddings
            grad_cat = grad_hidden @ W1.T
        else:
            # W, b
            grads[-2] = cat.T @ grad_logits + L2 * W
            grads[-1] = grad_logits.sum(axis=0)
            grad_cat = grad_logits @ W.T

        # distribute grad_cat to embeddings
        for j in range(d):
            gj = grad_cat[:, j*D:(j+1)*D]
            np.add.at(grads[j], Xb[:, j], gj)
            grads[j] += L2 * emb[j]
        return grads

    def evaluate(Xe, ye):
        correct = 0
        for s in range(0, len(Xe), BATCH):
            xb = Xe[s:s+BATCH]
            yb = ye[s:s+BATCH]
            logits, _ = forward(xb) if not H else (forward(xb)[0], None)
            pred = logits.argmax(axis=1)
            correct += (pred == yb).sum()
        return correct / len(Xe)

    # Training with early stopping on validation
    best_val = -1.0
    best_epoch = 0
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for epoch in range(max_epochs):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s+BATCH]
            grads = loss_and_grad(X_tr[idx], y_tr[idx])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        val_acc = evaluate(X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1

    # Retrain on full data for best_epoch epochs
    rng = np.random.default_rng(SEED)  # reset for reproducibility
    # re-initialize params
    emb = [(rng.uniform(-0.01, 0.01, (256, D))).astype(np.float32) for _ in range(d)]
    if H:
        W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = emb + [W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = emb + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            grads = loss_and_grad(X[idx], y[idx])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights.bin
    # Magic: 0x454D424C ('EMBL' little-endian)
    # Format: int32 magic, d, D, H, C; then embeddings[d][256][D] float32;
    # if H: W1[d*D][H], b1[H], W2[H][C], b2[C] else W[d*D][C], b[C]
    out = bytearray()
    out += struct.pack("<5i", 0x454D424C, d, D, H, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header\n"); return 3; }

    // Expected weight size
    long expected = 20;
    expected += (long)d * 256 * D * 4;  // embeddings
    if (H) {
        expected += (long)d * D * H * 4;  // W1
        expected += (long)H * 4;          // b1
        expected += (long)H * C * 4;      // W2
        expected += (long)C * 4;          // b2
    } else {
        expected += (long)d * D * C * 4;  // W
        expected += (long)C * 4;          // b
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p;  // [d][256][D]
    p += (long)d * 256 * D;
    const float *W1 = NULL, *b1 = NULL, *W2 = NULL, *b2 = NULL, *W = NULL, *b = NULL;
    if (H) {
        W1 = p; p += (long)d * D * H;
        b1 = p; p += H;
        W2 = p; p += (long)H * C;
        b2 = p;
    } else {
        W = p; p += (long)d * D * C;
        b = p;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *cat = malloc(sizeof(float) * d * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            float *dst = cat + j * D;
            memcpy(dst, src, sizeof(float) * D);
        }

        if (H) {
            // hidden = ReLU(cat @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int cd = 0; cd < d * D; cd++) sum += cat[cd] * W1[(long)cd * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = cat @ W + b
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int cd = 0; cd < d * D; cd++) sum += cat[cd] * W[(long)cd * C + c];
                logits[c] = sum;
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```