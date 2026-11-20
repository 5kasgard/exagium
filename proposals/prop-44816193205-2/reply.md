```hypothesis
Family: label-posmlp (Positional Embedding MLP). Learns token embeddings (256×E) and positional embeddings (16×E), sums them per position, concatenates (16E), then one ReLU hidden layer (H) and linear output (65). Captures sequential structure via positional embeddings and non-linear interactions. Expected >35% accuracy at ~50k insn/item (vs label-linear 33.75% at 7k). Fails if accuracy ≤33.75% or insn >100k.
```
```python
import json, math, struct, sys
import numpy as np

CONFIGS = [
    {"E": 16, "H": 64, "lr": 3e-3, "L2": 1e-4, "epochs": 10},
    {"E": 32, "H": 128, "lr": 3e-3, "L2": 1e-4, "epochs": 10},
    {"E": 32, "H": 64, "lr": 1e-3, "L2": 1e-3, "epochs": 10},
    {"E": 64, "H": 128, "lr": 1e-3, "L2": 1e-4, "epochs": 10},
]

SEED, BATCH, SAMPLES, MAX_EPOCHS = 1, 64, 600000, 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(E, H, C, rng):
    token_emb = rng.normal(0, 0.01, (256, E)).astype(np.float32)
    pos_emb = rng.normal(0, 0.01, (16, E)).astype(np.float32)
    W1 = rng.normal(0, np.sqrt(2.0 / (16 * E)), (16 * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return token_emb, pos_emb, W1, b1, W2, b2

def forward(token_emb, pos_emb, W1, b1, W2, b2, X):
    emb = token_emb[X] + pos_emb[None, :, :]
    flat = emb.reshape(X.shape[0], -1)
    h = flat @ W1 + b1
    np.maximum(h, 0, out=h)
    logits = h @ W2 + b2
    return logits, h, flat

def loss_and_grad(logits, y, h, flat, token_emb, pos_emb, W1, b1, W2, b2, X, L2):
    B = X.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    probs /= B
    dW2 = h.T @ probs + L2 * W2
    db2 = probs.sum(axis=0)
    dh = probs @ W2.T
    dh *= (h > 0)
    dW1 = flat.T @ dh + L2 * W1
    db1 = dh.sum(axis=0)
    dflat = dh @ W1.T
    demb = dflat.reshape(B, 16, -1)
    dtoken_emb = np.zeros_like(token_emb)
    for i in range(16):
        np.add.at(dtoken_emb, X[:, i], demb[:, i])
    dpos_emb = demb.sum(axis=0)
    return dtoken_emb, dpos_emb, dW1, db1, dW2, db2

def train_epoch(token_emb, pos_emb, W1, b1, W2, b2, X, y, lr, L2, batch_size, rng, adam_state, t):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        xb, yb = X[idx], y[idx]
        logits, h, flat = forward(token_emb, pos_emb, W1, b1, W2, b2, xb)
        grads = loss_and_grad(logits, yb, h, flat, token_emb, pos_emb, W1, b1, W2, b2, xb, L2)
        params = [token_emb, pos_emb, W1, b1, W2, b2]
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = adam_state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        t += 1
    return t

def evaluate(token_emb, pos_emb, W1, b1, W2, b2, X, y):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb, yb = X[s:s + 1024], y[s:s + 1024]
        logits, _, _ = forward(token_emb, pos_emb, W1, b1, W2, b2, xb)
        correct += (logits.argmax(axis=1) == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    best_acc = -1
    best_config = None
    best_params = None

    for ci, cfg in enumerate(CONFIGS):
        rng = np.random.default_rng(SEED + ci)
        params = init_params(cfg["E"], cfg["H"], C, rng)
        adam_state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        for _ in range(cfg["epochs"]):
            t = train_epoch(*params, X_train, y_train, cfg["lr"], cfg["L2"], BATCH, rng, adam_state, t)
        acc = evaluate(*params, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_config = cfg
            best_params = [p.copy() for p in params]
        print(f"Config {ci}: E={cfg['E']}, H={cfg['H']}, lr={cfg['lr']}, L2={cfg['L2']}, tune_acc={acc:.4f}", file=sys.stderr)

    print(f"Best: E={best_config['E']}, H={best_config['H']}, lr={best_config['lr']}, L2={best_config['L2']}, acc={best_acc:.4f}", file=sys.stderr)

    E, H = best_config["E"], best_config["H"]
    rng = np.random.default_rng(SEED + 999)
    params = init_params(E, H, C, rng)
    adam_state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        t = train_epoch(*params, X, y, best_config["lr"], best_config["L2"], BATCH, rng, adam_state, t)

    magic = 0x504D4C31
    out = struct.pack("<5i", magic, d, E, H, C)
    for p in params:
        out += p.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20 || hd[0] != 0x504D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * (256L * E + (long)d * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf