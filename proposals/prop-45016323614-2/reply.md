```hypothesis
Family: label-emb (embedding + shallow MLP). Each of the 16 context bytes gets a learned dense embedding (size E), concatenated, then passed through an optional ReLU hidden layer (size H) to 65 logits. Embeddings capture character similarities and reduce dimensionality vs one-hot, improving accuracy over label-linear. Hyperparameters (E, H) are chosen on the last 20% of train.in, then retrained on all data. Failure: dev accuracy ≤ label-linear (0.3375) or inference cost significantly higher.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, rng, epochs, n_samples):
    n = X.shape[0]
    # Initialize parameters
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    if H > 0:
        W1 = rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W_out = rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            # Forward: concatenate embeddings
            concat = np.zeros((len(yb), d*E), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
            if H > 0:
                hidden = concat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2
            else:
                logits = concat @ W_out + b_out
            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            # Backward
            if H > 0:
                grad_W2 = hidden_relu.T @ grad_logits
                grad_b2 = grad_logits.sum(0)
                grad_hidden_relu = grad_logits @ W2.T
                grad_hidden = grad_hidden_relu * (hidden > 0)
                grad_W1 = concat.T @ grad_hidden
                grad_b1 = grad_hidden.sum(0)
                grad_concat = grad_hidden @ W1.T
            else:
                grad_W_out = concat.T @ grad_logits
                grad_b_out = grad_logits.sum(0)
                grad_concat = grad_logits @ W_out.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_concat[:, j*E:(j+1)*E])
            if H > 0:
                grads = [grad_emb, grad_W1 + L2*W1, grad_b1, grad_W2 + L2*W2, grad_b2]
            else:
                grads = [grad_emb, grad_W_out + L2*W_out, grad_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, d, C, E, H, params, batch_size=1024):
    emb = params[0]
    n = X.shape[0]
    correct = 0
    for s in range(0, n, batch_size):
        xb = X[s:s+batch_size]
        yb = y[s:s+batch_size]
        concat = np.zeros((len(yb), d*E), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
        if H > 0:
            W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
            hidden = np.maximum(concat @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            W_out, b_out = params[1], params[2]
            logits = concat @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X_bytes[train_idx], y[train_idx]
    X_val, y_val = X_bytes[val_idx], y[val_idx]

    configs = [(16,0),(16,32),(16,64),(32,0),(32,32),(32,64),(64,0),(64,32),(64,64)]
    best_acc = -1
    best_config = None

    rng = np.random.default_rng(SEED)
    for E, H in configs:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X_train.shape[0]))
        params = train_model(X_train, y_train, d, C, E, H, rng, epochs, SAMPLES)
        acc = evaluate(X_val, y_val, d, C, E, H, params)
        print(f"E={E} H={H} val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_config = (E, H)

    print(f"Best: E={best_config[0]} H={best_config[1]} acc={best_acc:.4f}", file=sys.stderr)
    E, H = best_config

    # Retrain on full data with fresh RNG for determinism
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(X_bytes, y, d, C, E, H, rng, epochs, SAMPLES)

    # Save weights
    magic = 0x454D424C
    emb = params[0]
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    if H > 0:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W_out, b_out = params[1], params[2]
        out += W_out.astype("<f4").tobytes()
        out += b_out.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (H ? (long)d * E * H + H + (long)H * C + C : (long)d * E * C + C);
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    const float *W_out = W1;
    const float *b_out = b1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W_out[(long)idx * C + c];
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