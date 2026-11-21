```hypothesis
Family: label-embed (factorized linear with per-position character embeddings).
Why: The task predicts the next character from 16 bytes of context. A full linear model (label-linear) uses one-hot features per position, ignoring similarities between characters. By learning a small embedding (D=8) for each byte value at each position, we capture character similarities and positional interactions with far fewer parameters than a full linear layer (16*256*8 + 16*8*65 ≈ 41k floats vs 16*256*65 ≈ 266k). This should improve accuracy over label-linear (33.75%) while keeping inference fast (16 table lookups + one matrix-vector multiply). If dev accuracy does not exceed label-linear or inference cost grows disproportionately, the hypothesis fails.
```

```python
"""label-embed: factorized linear model with per-position character embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Exports: magic 0x454D4244 ('EMBD'), d, D, C; float32 emb[d][256][D], W_out[d*D][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter candidates for embedding dimension D
D_CANDIDATES = [4, 8, 16, 32]
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


def train_eval(emb, W_out, b_out, X, y, d, D, C):
    """Compute accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        # Build hidden
        hid = np.zeros((len(yb), d*D), dtype=np.float32)
        for j in range(d):
            hid[:, j*D:(j+1)*D] = emb[j, xb[:, j], :]
        logits = hid @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def sgd_step(emb, W_out, b_out, X, y, d, D, C, lr, l2, m_emb, v_emb, m_W, v_W, m_b, v_b, t):
    """One minibatch Adam step."""
    n = X.shape[0]
    # Forward
    hid = np.zeros((n, d*D), dtype=np.float32)
    for j in range(d):
        hid[:, j*D:(j+1)*D] = emb[j, X[:, j], :]
    logits = hid @ W_out + b_out
    # Softmax gradient
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # Gradients
    g_W = hid.T @ probs + l2 * W_out
    g_b = probs.sum(axis=0)
    g_hid = probs @ W_out.T  # (n, d*D)
    # Embedding gradients
    g_emb = np.zeros_like(emb)
    for j in range(d):
        # g_emb[j, byte, :] += sum over samples where X[:,j]==byte of g_hid[:, j*D:(j+1)*D]
        for byte in range(256):
            mask = (X[:, j] == byte)
            if mask.any():
                g_emb[j, byte, :] = g_hid[mask, j*D:(j+1)*D].sum(axis=0)
    g_emb += l2 * emb
    # Adam updates
    t += 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # Embeddings
    m_emb *= beta1; m_emb += (1-beta1) * g_emb
    v_emb *= beta2; v_emb += (1-beta2) * g_emb * g_emb
    m_emb_hat = m_emb / (1 - beta1**t)
    v_emb_hat = v_emb / (1 - beta2**t)
    emb -= lr * m_emb_hat / (np.sqrt(v_emb_hat) + eps)
    # W_out
    m_W *= beta1; m_W += (1-beta1) * g_W
    v_W *= beta2; v_W += (1-beta2) * g_W * g_W
    m_W_hat = m_W / (1 - beta1**t)
    v_W_hat = v_W / (1 - beta2**t)
    W_out -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
    # b_out
    m_b *= beta1; m_b += (1-beta1) * g_b
    v_b *= beta2; v_b += (1-beta2) * g_b * g_b
    m_b_hat = m_b / (1 - beta1**t)
    v_b_hat = v_b / (1 - beta2**t)
    b_out -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    return t


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    best_params = None

    for D in D_CANDIDATES:
        # Initialize
        emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256*D))).astype(np.float32)
        W_out = (rng.uniform(-1, 1, (d*D, C)) * math.sqrt(6.0 / (d*D))).astype(np.float32)
        b_out = np.zeros(C, dtype=np.float32)
        m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
        m_W = np.zeros_like(W_out); v_W = np.zeros_like(W_out)
        m_b = np.zeros_like(b_out); v_b = np.zeros_like(b_out)
        t = 0

        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s+BATCH]
                t = sgd_step(emb, W_out, b_out, X_tr[idx], y_tr[idx], d, D, C,
                             LR, L2, m_emb, v_emb, m_W, v_W, m_b, v_b, t)
        # Evaluate on validation
        acc = train_eval(emb, W_out, b_out, X_val, y_val, d, D, C)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_params = (emb.copy(), W_out.copy(), b_out.copy())

    # Retrain best on full training set
    D = best_D
    emb, W_out, b_out = best_params
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W_out); v_W = np.zeros_like(W_out)
    m_b = np.zeros_like(b_out); v_b = np.zeros_like(b_out)
    t = 0
    n_tr = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            t = sgd_step(emb, W_out, b_out, X[idx], y[idx], d, D, C,
                         LR, L2, m_emb, v_emb, m_W, v_W, m_b, v_b, t)

    # Export weights
    # Magic 'EMBD' = 0x454D4244
    out = struct.pack("<4i", 0x454D4244, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D4244, int32 d, D, C; float32 emb[d][256][D], W_out[d*D][C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || D < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * D;
    const float *b_out = W_out + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hid = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build hidden vector: concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(hid + j * D, e, sizeof(float) * D);
        }

        // Compute scores = hid @ W_out + b_out
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int h = 0; h < d * D; h++) {
            float hv = hid[h];
            if (hv == 0.0f) continue;
            const float *w = W_out + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```