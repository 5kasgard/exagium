```hypothesis
Family: label-embmlp (embedding + one hidden ReLU layer). 
Why: The task predicts the next character from 16 bytes of context. Linear model (label-linear) uses one-hot per position (4096 features) and achieves 33.75% accuracy. An embedding layer reduces dimensionality (16*8=128 dense features) and allows sharing statistical strength across byte values, while a hidden ReLU layer (64 units) captures pairwise interactions between positions. This architecture is more expressive than linear but far smaller and faster than label-mlp16 (which uses one-hot + 16 hidden units). Expected dev accuracy >35% with inference cost ~2x linear.
Failure mode: If embeddings do not generalize (bytes are not semantically similar) or hidden layer is too small, accuracy may match or fall below linear.
```

```python
"""label-embmlp: Embedding + one hidden ReLU layer for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d*256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
E = 8          # embedding size per position
H = 64         # hidden layer size
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, X_tune = X[:train_size], X[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 0.01, (d * 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    pos_offsets = np.arange(d) * 256

    def forward(Xb):
        idx = Xb + pos_offsets
        emb_batch = emb[idx].reshape(Xb.shape[0], d * E)
        h_pre = emb_batch @ W1 + b1
        h_act = np.maximum(h_pre, 0)
        logits = h_act @ W2 + b2
        return emb_batch, h_pre, h_act, logits

    def loss_grad(logits, yb):
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return probs / len(yb)

    best_tune_acc = -1.0
    best_params = None
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(train_size)
        for s in range(0, train_size, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            emb_batch, h_pre, h_act, logits = forward(Xb)
            grad_out = loss_grad(logits, yb)

            grad_W2 = h_act.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)

            grad_h_act = grad_out @ W2.T
            grad_h_pre = grad_h_act * (h_pre > 0)

            grad_W1 = emb_batch.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)

            grad_emb_flat = (grad_h_pre @ W1.T).reshape(-1, d, E)
            idx_flat = (Xb + pos_offsets).reshape(-1)
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, idx_flat, grad_emb_flat.reshape(-1, E))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        _, _, _, logits_tune = forward(X_tune)
        pred_tune = logits_tune.argmax(axis=1)
        tune_acc = (pred_tune == y_tune).mean()
        if tune_acc > best_tune_acc:
            best_tune_acc = tune_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]

    print(f"Best tune epoch: {best_epoch}, acc: {best_tune_acc:.4f}", file=sys.stderr)

    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 0.01, (d * 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            emb_batch, h_pre, h_act, logits = forward(Xb)
            grad_out = loss_grad(logits, yb)

            grad_W2 = h_act.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)

            grad_h_act = grad_out @ W2.T
            grad_h_pre = grad_h_act * (h_pre > 0)

            grad_W1 = emb_batch.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)

            grad_emb_flat = (grad_h_pre @ W1.T).reshape(-1, d, E)
            idx_flat = (Xb + pos_offsets).reshape(-1)
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, idx_flat, grad_emb_flat.reshape(-1, E))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x454D424C
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
    out += params[0].astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()
    out += params[2].astype("<f4").tobytes()
    out += params[3].astype("<f4").tobytes()
    out += params[4].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d*256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + j * E, e, E * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) {
                sum += emb_concat[f] * W1[(long)f * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(emb_concat);
    free(hidden);
    free(logits);
    return 0;
}
```