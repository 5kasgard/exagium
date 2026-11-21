```hypothesis
Family: label-embed (position-specific embeddings + optional one-hidden-layer MLP).
Why: Character-level language modeling benefits from dense representations of bytes; embeddings capture similarity between characters and reduce parameters vs. one-hot linear (266k → ~40-80k). A small hidden layer adds non-linearity for context interactions. Tuning embedding size D and hidden size H on the 20% validation split should beat label-linear's 0.3375 accuracy at lower inference cost (fewer multiply-adds).
Failure: If best validation accuracy ≤ 0.3375 or inference cost exceeds label-linear.
```
```python
"""label-embed: position-specific embeddings + linear/MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

# Hyperparameter grid to search on the tuning split (last 20% of train.in)
# (embedding_dim D, hidden_size H). H=0 means linear on embeddings.
HP_GRID = [
    (8, 0), (16, 0), (32, 0), (64, 0),
    (8, 32), (16, 32), (32, 32),
]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# For hyperparameter search we use fewer epochs to save time
SEARCH_EPOCHS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def split_data(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    # last 20% in file order
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    return X_train, y_train, X_val, y_val

def init_params(d, D, H, C, rng):
    # embeddings: d x 256 x D
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d*D, H)) * math.sqrt(6.0 / (d*D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (d*D, C)) * math.sqrt(6.0 / (d*D))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [E, W, b]
    return params

def forward(params, X, d, D, H, C):
    n = X.shape[0]
    E = params[0]
    # gather embeddings: for each position j, shape (n, D)
    emb_list = [E[j, X[:, j]] for j in range(d)]
    x_emb = np.concatenate(emb_list, axis=1).astype(np.float32)  # (n, d*D)
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
        hidden = x_emb @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU in-place
        logits = hidden @ W2 + b2
    else:
        W, b = params[1], params[2]
        logits = x_emb @ W + b
    return logits, x_emb

def loss_and_grad(params, X, y, d, D, H, C):
    n = X.shape[0]
    logits, x_emb = forward(params, X, d, D, H, C)
    # softmax cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    loss = -np.log(np.maximum(1e-12, probs[np.arange(n), y] + 1.0/n)).mean()  # approximate
    # gradients
    if H:
        E, W1, b1, W2, b2 = params
        # dlogits = probs
        dW2 = x_emb.T @ probs  # wait, need hidden
        # recompute hidden
        hidden = x_emb @ W1 + b1
        hidden[hidden < 0] = 0
        dW2 = hidden.T @ probs
        db2 = probs.sum(axis=0)
        dhidden = probs @ W2.T
        dhidden[hidden == 0] = 0
        dW1 = x_emb.T @ dhidden
        db1 = dhidden.sum(axis=0)
        dx_emb = dhidden @ W1.T
    else:
        E, W, b = params
        dW = x_emb.T @ probs
        db = probs.sum(axis=0)
        dx_emb = probs @ W.T
    # L2 regularization gradients
    if H:
        dW2 += L2 * W2
        dW1 += L2 * W1
    else:
        dW += L2 * W
    # embedding gradients
    dE = np.zeros_like(E)
    for j in range(d):
        idx = X[:, j]
        grad_j = dx_emb[:, j*D:(j+1)*D]
        np.add.at(dE[j], idx, grad_j)
    dE += L2 * E
    if H:
        grads = [dE, dW1, db1, dW2, db2]
    else:
        grads = [dE, dW, db]
    return loss, grads

def train_model(X, y, d, D, H, C, epochs, rng):
    params = init_params(d, D, H, C, rng)
    n = X.shape[0]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            loss, grads = loss_and_grad(params, Xb, yb, d, D, H, C)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def accuracy(params, X, y, d, D, H, C):
    logits, _ = forward(params, X, d, D, H, C)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def pack_weights(params, d, D, H, C):
    E = params[0]
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
        magic = 0x454D4244  # 'EMBD'
        header = struct.pack("<5i", magic, d, D, H, C)
        body = E.astype("<f4").tobytes()
        body += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        body += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = params[1], params[2]
        magic = 0x454D4244
        header = struct.pack("<5i", magic, d, D, 0, C)
        body = E.astype("<f4").tobytes()
        body += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # split for hyperparameter tuning
    X_train, y_train, X_val, y_val = split_data(X, y, 0.2)

    best_acc = -1.0
    best_params = None
    best_D, best_H = None, None

    for D, H in HP_GRID:
        print(f"Trying D={D}, H={H}...", file=sys.stderr)
        params = train_model(X_train, y_train, d, D, H, C, SEARCH_EPOCHS, rng)
        acc = accuracy(params, X_val, y_val, d, D, H, C)
        print(f"  val acc = {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_D, best_H = D, H

    print(f"Best: D={best_D}, H={best_H}, val acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data
    final_params = train_model(X, y, d, best_D, best_H, C, MAX_EPOCHS, rng)
    # write weights
    out_bytes = pack_weights(final_params, d, best_D, best_H, C)
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights format: magic(0x454D4244), d, D, H, C (int32 each).
 * Then embeddings[d][256][D] (float32), then if H>0: W1[d*D][H], b1[H], W2[H][C], b2[C]; else W[d*D][C], b[C]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }
    // expected size
    long expected = 20;
    expected += (long)d * 256 * D * 4;
    if (H) {
        expected += (long)d * D * H * 4 + H * 4;
        expected += (long)H * C * 4 + C * 4;
    } else {
        expected += (long)d * D * C * 4 + C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    const float *W = W1; // reuse pointer when H=0
    const float *b = b1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // gather embeddings
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            memcpy(emb + j * D, e, D * sizeof(float));
        }
        if (H) {
            // hidden = relu(emb @ W1 + b1)
            for (int h = 0; h < H; h++) hidden[h] = b1[h];
            for (int idx = 0; idx < d * D; idx++) {
                float v = emb[idx];
                if (v == 0.0f) continue;
                const float *w = W1 + (long)idx * H;
                for (int h = 0; h < H; h++) hidden[h] += v * w[h];
            }
            for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
            // scores = hidden @ W2 + b2
            memcpy(scores, b2, C * sizeof(float));
            for (int h = 0; h < H; h++) {
                float v = hidden[h];
                if (v == 0.0f) continue;
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        } else {
            // scores = emb @ W + b
            memcpy(scores, b, C * sizeof(float));
            for (int idx = 0; idx < d * D; idx++) {
                float v = emb[idx];
                if (v == 0.0f) continue;
                const float *w = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```