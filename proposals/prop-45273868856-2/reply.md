```hypothesis
Family: label-embed-mlp. Learns a per-position byte embedding (dim E), sums them, then passes through an optional ReLU hidden layer (dim H) to 65-class output. This captures position-wise interactions more efficiently than one-hot linear models and adds non-linearity. Hyperparameters (E, H, lr, L2) are chosen on the last 20% of train.in. Expect to beat label-linear (0.3375) on dev; if accuracy ≤ 0.34 the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS_GRID = 30
MAX_EPOCHS_FULL = 50
GRID = {
    'E': [16, 32, 64],
    'H': [0, 32, 64],
    'lr': [1e-3, 3e-3],
    'L2': [1e-4, 1e-5],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def forward(X, params, E, H, C):
    # X: (n, 16) uint8
    n, d = X.shape
    emb = params['emb']  # (d, 256, E)
    # Sum embeddings
    acc = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        acc += emb[j, X[:, j], :]
    if H > 0:
        W1, b1 = params['W1'], params['b1']  # (E, H), (H,)
        W2, b2 = params['W2'], params['b2']  # (H, C), (C,)
        h = acc @ W1 + b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
    else:
        Wout, bout = params['Wout'], params['bout']  # (E, C), (C,)
        logits = acc @ Wout + bout
    return logits

def loss_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # gradient w.r.t logits

def train_model(X_train, y_train, X_val, y_val, E, H, C, lr, L2, epochs):
    n, d = X_train.shape
    rng = np.random.default_rng(SEED)
    # Initialize parameters
    emb = (rng.normal(0, 1/math.sqrt(E), (d, 256, E))).astype(np.float32)
    if H > 0:
        W1 = (rng.normal(0, math.sqrt(2.0/E), (E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = {'emb': emb, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        all_params = [emb, W1, b1, W2, b2]
    else:
        Wout = (rng.normal(0, math.sqrt(2.0/E), (E, C))).astype(np.float32)
        bout = np.zeros(C, np.float32)
        params = {'emb': emb, 'Wout': Wout, 'bout': bout}
        all_params = [emb, Wout, bout]
    # Adam state
    m = [np.zeros_like(p) for p in all_params]
    v = [np.zeros_like(p) for p in all_params]
    t = 0
    best_val = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_train[idx], y_train[idx]
            # Forward
            acc = np.zeros((len(yb), E), np.float32)
            for j in range(d):
                acc += emb[j, xb[:, j], :]
            if H > 0:
                h = acc @ W1 + b1
                h_relu = np.maximum(h, 0)
                logits = h_relu @ W2 + b2
            else:
                logits = acc @ Wout + bout
            # Backward
            g = loss_grad(logits, yb)  # (batch, C)
            if H > 0:
                # Output layer grads
                gW2 = h_relu.T @ g + L2 * W2
                gb2 = g.sum(0)
                # Hidden layer grads
                gh = g @ W2.T
                gh[h <= 0] = 0.0
                gW1 = acc.T @ gh + L2 * W1
                gb1 = gh.sum(0)
                # Embedding grads
                gacc = gh @ W1.T
                gemb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(gemb[j], xb[:, j], gacc)
                gemb += L2 * emb
                grads = [gemb, gW1, gb1, gW2, gb2]
            else:
                gWout = acc.T @ g + L2 * Wout
                gbout = g.sum(0)
                gacc = g @ Wout.T
                gemb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(gemb[j], xb[:, j], gacc)
                gemb += L2 * emb
                grads = [gemb, gWout, gbout]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(all_params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation
        val_logits = forward(X_val, params, E, H, C)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = {k: v.copy() for k, v in params.items()}
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_overall = -1.0
    best_hp = None
    best_params = None
    for E in GRID['E']:
        for H in GRID['H']:
            for lr in GRID['lr']:
                for L2 in GRID['L2']:
                    val_acc, params = train_model(X_train, y_train, X_val, y_val, E, H, C, lr, L2, MAX_EPOCHS_GRID)
                    if val_acc > best_overall:
                        best_overall = val_acc
                        best_hp = (E, H, lr, L2)
                        best_params = params
    # Retrain on full data
    E, H, lr, L2 = best_hp
    _, final_params = train_model(X, y, X, y, E, H, C, lr, L2, MAX_EPOCHS_FULL)
    # Save weights
    emb = final_params['emb']  # (d, 256, E)
    magic = 0x454D424C  # "LMBE"
    if H > 0:
        W1, b1 = final_params['W1'], final_params['b1']
        W2, b2 = final_params['W2'], final_params['b2']
        out = struct.pack("<5i", magic, d, E, H, C)
        out += emb.astype("<f4").tobytes()
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        Wout, bout = final_params['Wout'], final_params['bout']
        out = struct.pack("<5i", magic, d, E, 0, C)
        out += emb.astype("<f4").tobytes()
        out += Wout.astype("<f4").tobytes() + bout.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * E;
    size_t expected;
    if (H > 0) {
        expected = 20 + 4*emb_sz + 4*((size_t)E*H + H + (size_t)H*C + C);
    } else {
        expected = 20 + 4*emb_sz + 4*((size_t)E*C + C);
    }
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *Wout, *bout;
    if (H > 0) {
        W1 = emb + emb_sz;
        b1 = W1 + (size_t)E * H;
        W2 = b1 + H;
        b2 = W2 + (size_t)H * C;
    } else {
        Wout = emb + emb_sz;
        bout = Wout + (size_t)E * C;
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *row = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += row[e];
        }
        if (H > 0) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += acc[e] * W1[(size_t)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = bout[c];
                for (int e = 0; e < E; e++) sum += acc[e] * Wout[(size_t)e * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```