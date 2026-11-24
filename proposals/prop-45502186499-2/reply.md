```hypothesis
The label-embedding family uses per-position byte embeddings summed into a fixed-size vector, followed by a linear output layer. This factorizes the full linear model (16×256×65 weights) into 16×256×D embeddings plus D×65 output weights, capturing positional similarities and reducing parameters. For character-level language modeling, nearby bytes often share predictive patterns; embeddings allow generalization across byte values. With D=16, the model has ~66k parameters (vs 266k for label-linear) and inference requires only 16×D + D×65 ≈ 1300 multiply-adds. If dev accuracy does not exceed 0.3375 (label-linear) or inference cost exceeds label-linear's ~7k Ir/item, the family is not beneficial.
```
```python
"""label-embedding: per-position byte embeddings summed + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split in train.py)
SEED = 1
BATCH = 64
EPOCHS = 30
L2 = 1e-4
# Grid for tuning split (last 20% of train.in)
EMBED_DIMS = [8, 16, 32]
LRS = [0.03, 0.1, 0.3]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    # Split: last 20% for tuning (by file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    def init_params(D):
        # embeddings: (d, 256, D)
        emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
        W = (rng.normal(0, 0.01, (C, D))).astype(np.float32)
        b = np.zeros(C, np.float32)
        return emb, W, b

    def forward(emb, W, b, Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # sum embeddings: (batch, D)
        summed = np.zeros((batch, emb.shape[2]), np.float32)
        for j in range(d):
            summed += emb[j, Xb[:, j]]
        logits = summed @ W.T + b
        return summed, logits

    def loss_and_grad(emb, W, b, Xb, yb):
        batch = Xb.shape[0]
        summed, logits = forward(emb, W, b, Xb)
        # softmax cross-entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # gradient of loss w.r.t logits
        grad_logits = probs
        grad_logits[np.arange(batch), yb] -= 1.0
        grad_logits /= batch
        # gradients
        # dL/dW = grad_logits.T @ summed  (C, D)
        grad_W = grad_logits.T @ summed
        grad_b = grad_logits.sum(axis=0)
        # dL/dsummed = grad_logits @ W  (batch, D)
        grad_summed = grad_logits @ W
        # dL/demb: for each position j, add grad_summed to emb[j, Xb[:, j]]
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            np.add.at(grad_emb[j], Xb[:, j], grad_summed)
        # L2 regularization
        grad_W += L2 * W
        grad_b += L2 * b
        grad_emb += L2 * emb
        # loss value
        loss = -np.log(probs[np.arange(batch), yb] + 1e-12).mean()
        loss += 0.5 * L2 * (np.sum(emb*emb) + np.sum(W*W) + np.sum(b*b))
        return loss, grad_emb, grad_W, grad_b

    def evaluate(emb, W, b, X_eval, y_eval):
        correct = 0
        for s in range(0, len(X_eval), BATCH):
            Xb = X_eval[s:s+BATCH]
            yb = y_eval[s:s+BATCH]
            _, logits = forward(emb, W, b, Xb)
            pred = logits.argmax(axis=1)
            correct += (pred == yb).sum()
        return correct / len(X_eval)

    def train_config(D, lr):
        emb, W, b = init_params(D)
        # Adam state
        m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_b, v_b = np.zeros_like(b), np.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        order = np.arange(n_train)
        for epoch in range(EPOCHS):
            rng.shuffle(order)
            for s in range(0, n_train, BATCH):
                idx = order[s:s+BATCH]
                Xb, yb = X_train[idx], y_train[idx]
                loss, g_emb, g_W, g_b = loss_and_grad(emb, W, b, Xb, yb)
                t += 1
                # Adam update
                for param, grad, m, v in [(emb, g_emb, m_emb, v_emb),
                                          (W, g_W, m_W, v_W),
                                          (b, g_b, m_b, v_b)]:
                    m *= beta1; m += (1 - beta1) * grad
                    v *= beta2; v += (1 - beta2) * grad * grad
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return emb, W, b

    # Grid search on tuning split
    best_acc = -1.0
    best_D = EMBED_DIMS[0]
    best_lr = LRS[0]
    for D in EMBED_DIMS:
        for lr in LRS:
            emb, W, b = train_config(D, lr)
            acc = evaluate(emb, W, b, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_lr = lr

    # Retrain on full data with best config
    emb, W, b = train_config(best_D, best_lr)

    # Export weights
    # Format: magic(0x454D4244), d, D, C; then embeddings (d*256*D floats), W (C*D floats), b (C floats)
    magic = 0x454D4244  # 'EMBD' in little-endian
    out = struct.pack("<4i", magic, d, best_D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (argmax). */
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
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)C * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // zero summed
        for (int u = 0; u < D; u++) summed[u] = 0.0f;
        // accumulate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) summed[u] += e[u];
        }
        // compute logits = summed * W^T + b
        for (int c = 0; c < C; c++) {
            float val = b[c];
            const float *wc = W + (long)c * D;
            for (int u = 0; u < D; u++) val += summed[u] * wc[u];
            logits[c] = val;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```