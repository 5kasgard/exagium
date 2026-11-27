```hypothesis
We propose a one-hidden-layer MLP with per-position byte embeddings (family "label-mlp"). The 16-byte context has strong local interactions (n-grams) that a linear model cannot capture; a hidden layer of size 128–512 with ReLU can learn such features. We tune the hidden size on the last 20% of train.in (validation split) and retrain on all data. If validation accuracy does not exceed label-linear's 0.3375, the model fails to justify its extra cost.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter search space (tuned on validation split)
HIDDEN_SIZES = [128, 256, 512]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_model(H, X, y, epochs):
    """Train on given data for a fixed number of samples (SAMPLES). Returns parameters."""
    rng = np.random.default_rng(SEED)
    d, C = X.shape[1], 65
    # Embeddings: (d, 256, H)
    scale = math.sqrt(6.0 / (256 * H))
    E = (rng.uniform(-1, 1, (d, 256, H)) * scale).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]
            # Forward
            emb = E[np.arange(d), xb.T].transpose(1, 0, 2)   # (B, d, H)
            h = emb.sum(axis=1) + b1                         # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2                        # (B, C)
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)                    # (B, C)
            # Backward
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_b1 = grad_h.sum(axis=0)
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_h)
            grad_E += L2 * E
            grads = [grad_E, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(params, X, y):
    E, b1, W2, b2 = params
    d = X.shape[1]
    emb = E[np.arange(d), X.T].transpose(1, 0, 2)
    h = emb.sum(axis=1) + b1
    h_relu = np.maximum(h, 0)
    logits = h_relu @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    val_start = int(n_all * 0.8)
    X_train, X_val = X_all[:val_start], X_all[val_start:]
    y_train, y_val = y_all[:val_start], y_all[val_start:]

    # Choose hidden size on validation split
    epochs_train = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    best_H, best_acc, best_params = None, -1, None
    for H in HIDDEN_SIZES:
        params = train_model(H, X_train, y_train, epochs_train)
        acc = evaluate(params, X_val, y_val)
        if acc > best_acc:
            best_acc, best_H, best_params = acc, H, params
    print(f"Selected H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    final_params = train_model(best_H, X_all, y_all, epochs_full)
    E, b1, W2, b2 = final_params

    # Write weights: magic, d, H, C, then E (d*256*H), b1 (H), W2 (H*C), b2 (C) as little-endian float32
    magic = 0x4D4C5031  # "MLP1"
    d, C = X_all.shape[1], 65
    out = struct.pack("<4i", magic, d, best_H, C)
    out += E.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* MLP with per-position byte embeddings and one ReLU hidden layer.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *out_len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *out_len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long wlen, xlen = 0;
    long N = atol(argv[3]);
    unsigned char *wbuf = slurp(argv[1], &wlen);
    unsigned char *xbuf = N ? slurp(argv[2], &xlen) : NULL;

    int32_t *hdr = (int32_t *)wbuf;
    if (wlen < 16 || hdr[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], H = hdr[2], C = hdr[3];
    if (d != 16 || C != 65 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected_floats = (size_t)d * 256 * H + H + (size_t)H * C + C;
    if (wlen != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(wbuf + 16);
    float *E = ptr; ptr += (size_t)d * 256 * H;
    float *b1 = ptr; ptr += H;
    float *W2 = ptr; ptr += (size_t)H * C;
    float *b2 = ptr;

    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    long pos = 0;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(xbuf, xlen, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = E + ((size_t)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += e[h];
        }
        for (int h = 0; h < H; h++) {
            float v = acc[h];
            acc[h] = v > 0.0f ? v : 0.0f;   // ReLU in-place
        }
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w = W2 + (size_t)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(scores); free(wbuf); free(xbuf);
    return 0;
}
```