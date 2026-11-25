```hypothesis
We propose "label-embmlp": per-position byte embeddings concatenated and passed through a one-hidden-layer ReLU MLP.
This captures non-linear interactions between positions while keeping inference fast (embedding lookup + two matvecs).
Tuning embedding size (D) and hidden size (H) on the 20% tuning split should beat label-linear's 0.3375 accuracy
with CPU cost around 10k-15k Ir/item. Failure would show the task is mostly linear or our capacity is too low.
```
```python
import json, math, struct, sys
import numpy as np

MAGIC = 0x454D424C  # 'EMBL'
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def evaluate(X, y, d, C, emb, W1, b1, W2, b2):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        emb_vec = emb[np.arange(d), xb.T].transpose(1,0,2)
        flat = emb_vec.reshape(len(xb), -1)
        h = flat @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_model(X, y, d, C, D, H, epochs, rng):
    n = X.shape[0]
    # Initialize
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0/(d*D)), size=(d*D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    # Adam state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]; yb = y[idx]
            # Forward
            emb_vec = emb[np.arange(d), xb.T].transpose(1,0,2)  # (batch, d, D)
            flat = emb_vec.reshape(len(xb), -1)
            h = flat @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2
            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)
            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_flat = grad_h @ W1.T
            grad_emb_vec = grad_flat.reshape(len(xb), d, D)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_vec[:, j, :])
            grad_emb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W1, grad_W1, m_W1, v_W1),
                (b1, grad_b1, m_b1, v_b1),
                (W2, grad_W2, m_W2, v_W2),
                (b2, grad_b2, m_b2, v_b2)
            ]:
                m *= beta1; m += (1-beta1)*grad
                v *= beta2; v += (1-beta2)*grad*grad
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
    return emb, W1, b1, W2, b2

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    # Hyperparameter grid (chosen on tuning split)
    configs = [
        (16, 32), (16, 64), (16, 128),
        (32, 32), (32, 64), (32, 128),
        (64, 32), (64, 64), (64, 128),
    ]
    best_acc = -1
    best_cfg = None
    best_weights = None
    for D, H in configs:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        rng = np.random.default_rng(SEED + D*1000 + H)
        emb, W1, b1, W2, b2 = train_model(X_tr, y_tr, d, C, D, H, epochs, rng)
        acc = evaluate(X_tune, y_tune, d, C, emb, W1, b1, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (D, H)
            best_weights = (emb, W1, b1, W2, b2)
    print(f"Best tuning config: D={best_cfg[0]}, H={best_cfg[1]}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, H = best_cfg
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED + D*1000 + H + 999)
    emb, W1, b1, W2, b2 = train_model(X, y, d, C, D, H, epochs, rng)

    # Write weights.bin
    out = struct.pack("<5i", MAGIC, d, D, H, C)
    out += emb.astype("<f4").tobytes()
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
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20L
        + (long)d * 256 * D * 4
        + (long)d * D * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = (float *)malloc(sizeof(float) * d * D);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        memset(flat, 0, sizeof(float) * d * D);
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            float *dst = flat + (long)j * D;
            for (int kk = 0; kk < D; kk++) dst[kk] += src[kk];
        }
        // Hidden layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h; // W1 is (d*D) x H, column-major? Actually row-major: W1[flat_idx * H + h]
            for (int fd = 0; fd < d*D; fd++) sum += flat[fd] * w[fd * H];
            hidden[h] = sum > 0 ? sum : 0;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```