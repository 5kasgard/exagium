```hypothesis
Family: label-embnet (embedding MLP). Uses per-position byte embeddings (dim D) concatenated, one ReLU hidden layer (size H), then output. Captures positional interactions better than linear one-hot, with fewer parameters than full one-hot MLP. Tuned on last 20% of train. If dev accuracy <= label-linear (0.3375) or instructions/item > 2x label-linear, it fails.
```

```python
"""label-embnet: per-position embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all train."""
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning
GRID = [
    {"D": 8, "H": 32, "lr": 1e-3},
    {"D": 8, "H": 64, "lr": 1e-3},
    {"D": 16, "H": 32, "lr": 1e-3},
    {"D": 16, "H": 64, "lr": 1e-3},
]
SEED = 1
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def load_data(in_path, key_path, d):
    X_rec = records(in_path)
    y_rec = records(key_path)
    X = np.frombuffer(b"".join(X_rec), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_rec], dtype=np.int64)
    return X, y

def init_params(d, D, H, C, rng):
    embeddings = [rng.normal(0, 1/math.sqrt(D), (256, D)).astype(np.float32) for _ in range(d)]
    W1 = rng.normal(0, math.sqrt(2.0/(d*D)), (d*D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return embeddings, W1, b1, W2, b2

def forward(X_batch, embeddings, W1, b1, W2, b2, d, D):
    batch = X_batch.shape[0]
    embs = [embeddings[j][X_batch[:, j]] for j in range(d)]
    x = np.concatenate(embs, axis=1)
    hidden_pre = x @ W1 + b1
    hidden = np.maximum(hidden_pre, 0)
    logits = hidden @ W2 + b2
    return x, hidden_pre, hidden, logits

def loss_grad(logits, y_batch, hidden, hidden_pre, x, embeddings, W1, W2, d, D, L2):
    batch = logits.shape[0]
    # softmax cross-entropy gradient
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    dlogits = probs
    dlogits[np.arange(batch), y_batch] -= 1
    dlogits /= batch

    dW2 = hidden.T @ dlogits + L2 * W2
    db2 = dlogits.sum(axis=0)

    dhidden = dlogits @ W2.T
    dhidden_pre = dhidden * (hidden_pre > 0)

    dW1 = x.T @ dhidden_pre + L2 * W1
    db1 = dhidden_pre.sum(axis=0)

    dx = dhidden_pre @ W1.T
    grad_embeddings = [np.zeros_like(e) for e in embeddings]
    for j in range(d):
        np.add.at(grad_embeddings[j], X_batch[:, j], dx[:, j*D:(j+1)*D])
        grad_embeddings[j] += L2 * embeddings[j]
    return grad_embeddings, dW1, db1, dW2, db2

def adam_update(params, grads, states, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m[:] = beta1 * m + (1 - beta1) * g
        v[:] = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def evaluate(X, y, embeddings, W1, b1, W2, b2, d, D, batch=1024):
    correct = 0
    total = 0
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        yb = y[i:i+batch]
        _, _, _, logits = forward(xb, embeddings, W1, b1, W2, b2, d, D)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
        total += len(yb)
    return correct / total

def train_model(X_train, y_train, X_val, y_val, d, C, hp, rng):
    D, H, lr = hp["D"], hp["H"], hp["lr"]
    embeddings, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    params = []
    for e in embeddings: params.append(e)
    params += [W1, b1, W2, b2]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val = -1
    best_params = None
    patience = PATIENCE
    n = len(X_train)
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_train[idx], y_train[idx]
            x, hidden_pre, hidden, logits = forward(xb, embeddings, W1, b1, W2, b2, d, D)
            grads = loss_grad(logits, yb, hidden, hidden_pre, x, embeddings, W1, W2, d, D, L2)
            t += 1
            adam_update(params, grads, states, lr, t)
        val_acc = evaluate(X_val, y_val, embeddings, W1, b1, W2, b2, d, D)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break
    # restore best
    for p, bp in zip(params, best_params):
        p[:] = bp
    return best_val, params

def save_weights(path, d, D, H, C, embeddings, W1, b1, W2, b2):
    magic = 0x454D424E  # 'EMBN'
    header = struct.pack("<5i", magic, d, D, H, C)
    # embeddings: d * 256 * D floats
    emb_arr = np.stack(embeddings).astype("<f4")  # (d, 256, D)
    w1_arr = W1.astype("<f4")
    b1_arr = b1.astype("<f4")
    w2_arr = W2.astype("<f4")
    b2_arr = b2.astype("<f4")
    with open(path, "wb") as f:
        f.write(header)
        f.write(emb_arr.tobytes())
        f.write(w1_arr.tobytes())
        f.write(b1_arr.tobytes())
        f.write(w2_arr.tobytes())
        f.write(b2_arr.tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X, y = load_data(sys.argv[2], sys.argv[3], d)
    # split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_hp = None
    best_val = -1
    best_params = None

    for hp in GRID:
        # use a derived seed for each config to keep deterministic
        cfg_rng = np.random.default_rng(SEED + hash((hp["D"], hp["H"], hp["lr"])) % 2**32)
        val_acc, params = train_model(X_train, y_train, X_val, y_val, d, C, hp, cfg_rng)
        if val_acc > best_val:
            best_val = val_acc
            best_hp = hp
            best_params = params

    # final training on all data with best_hp
    final_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_model(X, y, X_val, y_val, d, C, best_hp, final_rng)
    embeddings, W1, b1, W2, b2 = final_params[:d], final_params[d], final_params[d+1], final_params[d+2], final_params[d+3]
    save_weights(sys.argv[4], d, best_hp["D"], best_hp["H"], C, embeddings, W1, b1, W2, b2)

if __name__ == "__main__":
    main()
```

```c
/* label-embnet inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x454D424E), d, D, H, C (int32 each),
 * then embeddings[d][256][D] (float32), W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424E) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected_floats = (long)d * 256 * D + (long)d * D * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden_pre = malloc(H * sizeof(float));
    float *output = malloc(C * sizeof(float));
    if (!hidden_pre || !output) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(hidden_pre, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            const float *e = emb + ((long)j * 256 + byte) * D;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int kk = 0; kk < D; kk++) {
                    sum += e[kk] * W1[((long)j * D + kk) * H + h];
                }
                hidden_pre[h] += sum;
            }
        }
        for (int h = 0; h < H; h++) hidden_pre[h] = hidden_pre[h] > 0.0f ? hidden_pre[h] : 0.0f;

        memcpy(output, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden_pre[h];
            for (int c = 0; c < C; c++) output[c] += hv * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(hidden_pre); free(output); free(x);
    return 0;
}
```