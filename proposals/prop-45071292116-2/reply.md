```hypothesis
We propose "label-emb-mlp": per-position byte embeddings (size 32) concatenated and fed to a one-hidden-layer ReLU MLP (size 128). This captures pairwise interactions between context positions while remaining parameter-efficient (~200K params vs linear's 266K) and fast at inference (fewer multiply-adds). If it fails to beat label-linear, it suggests the task is predominantly linear or that embeddings don't capture useful structure.
```

```python
"""label-emb-mlp: per-position embeddings + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all data."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split exploration)
EMBED_SIZE = 32
HIDDEN_SIZE = 128
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS_FULL = 20
EPOCHS_TUNE = 5
SEED = 1


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4 : i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, seed):
    rng = np.random.default_rng(seed)
    embed = [rng.normal(0, 0.01, size=(256, E)).astype(np.float32) for _ in range(d)]
    limit = np.sqrt(6.0 / (d * E + H))
    W1 = rng.uniform(-limit, limit, size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    limit2 = np.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-limit2, limit2, size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    return embed, W1, b1, W2, b2


def forward(X_batch, embed, W1, b1, W2, b2):
    B, d = X_batch.shape
    E = embed[0].shape[1]
    concat = np.empty((B, d * E), dtype=np.float32)
    for j in range(d):
        concat[:, j * E : (j + 1) * E] = embed[j][X_batch[:, j]]
    hidden_pre = concat @ W1 + b1
    hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
    logits = hidden @ W2 + b2
    return logits, hidden, hidden_pre, concat


def loss_and_grad(X_batch, y_batch, embed, W1, b1, W2, b2):
    B = X_batch.shape[0]
    logits, hidden, hidden_pre, concat = forward(X_batch, embed, W1, b1, W2, b2)
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    dlogits = probs
    dlogits[np.arange(B), y_batch] -= 1.0
    dlogits /= B
    dW2 = hidden.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    dhidden = dlogits @ W2.T
    dhidden[hidden_pre <= 0] = 0.0
    dW1 = concat.T @ dhidden
    db1 = np.sum(dhidden, axis=0)
    dconcat = dhidden @ W1.T
    d = X_batch.shape[1]
    E = embed[0].shape[1]
    dembed = [np.zeros_like(e) for e in embed]
    for j in range(d):
        idx = X_batch[:, j]
        grad_part = dconcat[:, j * E : (j + 1) * E]
        np.add.at(dembed[j], idx, grad_part)
    dW1 += L2 * W1
    dW2 += L2 * W2
    loss = -np.mean(np.log(probs[np.arange(B), y_batch] + 1e-12))
    return loss, (dembed, dW1, db1, dW2, db2)


def accuracy(X, y, embed, W1, b1, W2, b2, batch_size=1024):
    n = X.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        Xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits, _, _, _ = forward(Xb, embed, W1, b1, W2, b2)
        pred = np.argmax(logits, axis=1)
        correct += np.sum(pred == yb)
    return correct / n


def train_loop(X_train, y_train, X_val, y_val, d, C, E, H, lr, epochs, seed, init_params=None):
    rng = np.random.default_rng(seed)
    if init_params is None:
        embed, W1, b1, W2, b2 = init_params(d, E, H, C, seed)
    else:
        embed, W1, b1, W2, b2 = init_params
    all_params = [W1, b1, W2, b2] + embed
    m = [np.zeros_like(p) for p in all_params]
    v = [np.zeros_like(p) for p in all_params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X_train.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            Xb = X_train[idx]
            yb = y_train[idx]
            _, grads = loss_and_grad(Xb, yb, embed, W1, b1, W2, b2)
            dembed, dW1, db1, dW2, db2 = grads
            all_grads = [dW1, db1, dW2, db2] + dembed
            t += 1
            for i, (p, gr) in enumerate(zip(all_params, all_grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        val_acc = accuracy(X_val, y_val, embed, W1, b1, W2, b2)
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}", file=sys.stderr)
    return val_acc, (embed, W1, b1, W2, b2)


def train_full(X, y, d, C, E, H, lr, epochs, seed, init_params=None):
    rng = np.random.default_rng(seed)
    if init_params is None:
        embed, W1, b1, W2, b2 = init_params(d, E, H, C, seed)
    else:
        embed, W1, b1, W2, b2 = init_params
    all_params = [W1, b1, W2, b2] + embed
    m = [np.zeros_like(p) for p in all_params]
    v = [np.zeros_like(p) for p in all_params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            _, grads = loss_and_grad(Xb, yb, embed, W1, b1, W2, b2)
            dembed, dW1, db1, dW2, db2 = grads
            all_grads = [dW1, db1, dW2, db2] + dembed
            t += 1
            for i, (p, gr) in enumerate(zip(all_params, all_grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return embed, W1, b1, W2, b2


def save_weights(path, d, E, H, C, params):
    embed, W1, b1, W2, b2 = params
    magic = 0x4C424D45  # "LBME" little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in embed)
    w1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    w2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(header)
        f.write(emb_bytes)
        f.write(w1_bytes)
        f.write(b1_bytes)
        f.write(w2_bytes)
        f.write(b2_bytes)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_start = int(n * 0.8)
    X_train, X_val = X[:val_start], X[val_start:]
    y_train, y_val = y[:val_start], y[val_start:]

    best_acc = -1.0
    best_hp = None
    best_params = None
    for E in [16, 32]:
        for H in [64, 128]:
            for lr in [1e-3, 3e-3]:
                acc, params = train_loop(
                    X_train, y_train, X_val, y_val, d, C, E, H, lr, EPOCHS_TUNE, SEED
                )
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (E, H, lr)
                    best_params = params
    print(f"Best hp: E={best_hp[0]}, H={best_hp[1]}, lr={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    E, H, lr = best_hp
    final_params = train_full(X, y, d, C, E, H, lr, EPOCHS_FULL, SEED, best_params)
    save_weights(sys.argv[4], d, E, H, C, final_params)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: per-position embeddings + one ReLU hidden layer.
 * ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: one byte class index (0..64). */
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
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(hd + 5);
    float *embed = ptr; ptr += (long)d * 256 * E;
    float *W1 = ptr;    ptr += (long)d * E * H;
    float *b1 = ptr;    ptr += H;
    float *W2 = ptr;    ptr += (long)H * C;
    float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden_pre = malloc(H * sizeof(float));
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(hidden_pre, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *emb = embed + ((long)j * 256 + byte) * E;
            for (int e = 0; e < E; e++) {
                float ev = emb[e];
                if (ev == 0.0f) continue;
                const float *w1_row = W1 + ((long)j * E + e) * H;
                for (int u = 0; u < H; u++) hidden_pre[u] += ev * w1_row[u];
            }
        }
        for (int u = 0; u < H; u++) hidden[u] = hidden_pre[u] > 0.0f ? hidden_pre[u] : 0.0f;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(hidden_pre); free(hidden); free(logits);
    return 0;
}
```