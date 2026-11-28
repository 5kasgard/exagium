```hypothesis
Family: label-pemb-mlp (positional-embedding MLP). Uses shared token embeddings (256 x E) plus positional embeddings (16 x E), summed to a single E-vector, then one ReLU hidden layer (H) and linear output. Captures position-sensitive interactions with far fewer parameters than one-hot linear models. Should beat label-linear (0.3375) on dev because it models non-linear position-byte interactions. If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2x, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid to search on the tuning split (last 20% of train.in)
GRID = [
    {"E": 16, "H": 32, "lr": 3e-3, "epochs": 30},
    {"E": 32, "H": 64, "lr": 3e-3, "epochs": 30},
    {"E": 64, "H": 128, "lr": 1e-3, "epochs": 30},
    {"E": 32, "H": 64, "lr": 1e-3, "epochs": 40},
]
BATCH = 64
L2 = 1e-5
CLIP_NORM = 1.0

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    # last 20% in file order
    X_train, X_tune = X[:-tune_n], X[-tune_n:]
    y_train, y_tune = y[:-tune_n], y[-tune_n:]
    return X_train, y_train, X_tune, y_tune

def init_params(E, H, C, rng):
    # token_emb: 256 x E, pos_emb: 16 x E, W1: E x H, b1: H, W2: H x C, b2: C
    scale = math.sqrt(6.0 / (256 + E))
    token_emb = rng.uniform(-scale, scale, (256, E)).astype(np.float32)
    scale = math.sqrt(6.0 / (16 + E))
    pos_emb = rng.uniform(-scale, scale, (16, E)).astype(np.float32)
    scale = math.sqrt(6.0 / (E + H))
    W1 = rng.uniform(-scale, scale, (E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    scale = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale, scale, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [token_emb, pos_emb, W1, b1, W2, b2]

def forward(params, X, training=False):
    token_emb, pos_emb, W1, b1, W2, b2 = params
    batch = X.shape[0]
    # sum of positional embeddings (constant)
    pos_sum = pos_emb.sum(axis=0)  # (E,)
    emb_sum = np.tile(pos_sum, (batch, 1))  # (batch, E)
    for j in range(16):
        emb_sum += token_emb[X[:, j]]
    hidden = emb_sum @ W1 + b1
    hidden_relu = np.maximum(hidden, 0)
    logits = hidden_relu @ W2 + b2
    if training:
        return logits, hidden_relu, emb_sum
    else:
        return logits

def loss_and_grad(params, X, y):
    token_emb, pos_emb, W1, b1, W2, b2 = params
    batch = X.shape[0]
    logits, hidden_relu, emb_sum = forward(params, X, training=True)
    # softmax cross-entropy
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    probs /= batch
    loss = -np.log(np.exp(logits[np.arange(batch), y] - max_logits.ravel()) / exp_logits.sum(axis=1)).mean()
    # gradients
    dlogits = probs.astype(np.float32)  # (batch, C)
    dW2 = hidden_relu.T @ dlogits + L2 * W2
    db2 = dlogits.sum(axis=0)
    dhidden_relu = dlogits @ W2.T
    dhidden = dhidden_relu * (hidden_relu > 0)
    dW1 = emb_sum.T @ dhidden + L2 * W1
    db1 = dhidden.sum(axis=0)
    demb_sum = dhidden @ W1.T  # (batch, E)
    # token_emb gradients
    dtoken_emb = np.zeros_like(token_emb)
    for j in range(16):
        idx = X[:, j]
        np.add.at(dtoken_emb, idx, demb_sum)
    # pos_emb gradients: each position gets demb_sum added
    dpos_emb = np.tile(demb_sum.sum(axis=0), (16, 1))  # (16, E)
    # clip gradients
    grads = [dtoken_emb, dpos_emb, dW1, db1, dW2, db2]
    for g in grads:
        norm = np.linalg.norm(g)
        if norm > CLIP_NORM:
            g *= CLIP_NORM / norm
    return loss, grads

def train_model(X, y, params, epochs, lr):
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            loss, grads = loss_and_grad(params, Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(params, X, y):
    logits = forward(params, X, training=False)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    # split train/tune (last 20% in file order)
    X_train, y_train, X_tune, y_tune = split_train_tune(X, y, 0.2)
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_cfg = None
    for cfg in GRID:
        print(f"Trying cfg: {cfg}", file=sys.stderr)
        params = init_params(cfg["E"], cfg["H"], C, rng)
        params = train_model(X_train, y_train, params, cfg["epochs"], cfg["lr"])
        acc = evaluate(params, X_tune, y_tune)
        print(f"  tune acc: {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg
    print(f"Best cfg: {best_cfg} acc: {best_acc:.4f}", file=sys.stderr)
    # retrain on full data
    final_params = init_params(best_cfg["E"], best_cfg["H"], C, rng)
    final_params = train_model(X, y, final_params, best_cfg["epochs"], best_cfg["lr"])
    token_emb, pos_emb, W1, b1, W2, b2 = final_params
    E, H = best_cfg["E"], best_cfg["H"]
    d = 16
    # write weights: magic, E, H, C, d, then arrays
    magic = 0x50454D42  # 'PEMB'
    out = struct.pack("<5i", magic, E, H, C, d)
    out += token_emb.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x50454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int E = hd[1], H = hd[2], C = hd[3], d = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + 16L * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *token_emb = (const float *)(hd + 5);
    const float *pos_emb = token_emb + 256L * E;
    const float *W1 = pos_emb + 16L * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    // precompute sum of positional embeddings
    float *pos_sum = malloc(E * sizeof(float));
    for (int e = 0; e < E; e++) {
        float s = 0.0f;
        for (int j = 0; j < 16; j++) s += pos_emb[j * E + e];
        pos_sum[e] = s;
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(E * sizeof(float));
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // emb_sum = pos_sum
        memcpy(emb_sum, pos_sum, E * sizeof(float));
        // add token embeddings
        for (int j = 0; j < d; j++) {
            const float *te = token_emb + (long)xi[j] * E;
            for (int e = 0; e < E; e++) emb_sum[e] += te[e];
        }
        // hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```