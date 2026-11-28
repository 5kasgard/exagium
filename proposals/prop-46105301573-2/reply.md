```hypothesis
Family: label-emblinear (factorized linear with per-position embeddings).
Why: Character-level language modeling benefits from shared representations across similar characters (e.g., letters). A linear model on one-hot features ignores this; embeddings capture similarity and reduce parameters. The factorized form (embedding per position + linear classifier) adds only O(16*256*D + 65*16*D) params vs O(16*256*65) for linear, and inference cost is C*16*D multiply-adds (D=8 gives ~8k ops, still low).
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >2x.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid
D_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
EPOCHS_CHOICES = [10, 20]
BATCH_SIZE = 64
BASE_SEED = 12345
L2 = 1e-4  # weight decay

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, D, lr, epochs, rng):
    n, d = X.shape
    # Initialize embeddings (d, 256, D) and classifier weights (C, d, D)
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W = rng.normal(0, 0.01, size=(C, d, D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Gather embeddings: (B, d, D)
            emb_x = emb[np.arange(d), xb.T].transpose(1, 0, 2)

            # Logits: (B, C)
            logits = b + np.einsum('bpd,cpd->bc', emb_x, W)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B   # (B, C)

            # Gradients
            dW = np.einsum('bc,bpd->cpd', grad_logits, emb_x)  # (C, d, D)
            db = grad_logits.sum(axis=0)                       # (C,)
            demb_x = np.einsum('bc,cpd->bpd', grad_logits, W)  # (B, d, D)

            # Scatter demb_x into demb (d, 256, D)
            demb = np.zeros_like(emb)
            for p in range(d):
                bytes_p = xb[:, p]
                np.add.at(demb[p], bytes_p, demb_x[:, p, :])

            # L2 regularization
            dW += L2 * W
            demb += L2 * emb
            db += L2 * b

            # Adam update
            t += 1
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * demb
            v_emb = beta2 * v_emb + (1 - beta2) * demb * demb
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

            # W
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW * dW
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

            # b
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db * db
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return emb, W, b

def evaluate(X, y, emb, W, b):
    emb_x = emb[np.arange(16), X.T].transpose(1, 0, 2)
    logits = b + np.einsum('bpd,cpd->bc', emb_x, W)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_emb = best_W = best_b = None
    best_D = best_lr = best_epochs = None

    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCHS_CHOICES:
                # Deterministic seed per hyperparameter set
                seed = BASE_SEED + D * 10000 + int(lr * 10000) * 10 + epochs
                rng = np.random.default_rng(seed)
                emb, W, b = train_model(X_train, y_train, C, D, lr, epochs, rng)
                acc = evaluate(X_tune, y_tune, emb, W, b)
                if acc > best_acc:
                    best_acc = acc
                    best_D, best_lr, best_epochs = D, lr, epochs
                    best_emb, best_W, best_b = emb, W, b

    # Retrain on full data with best hyperparameters
    seed = BASE_SEED + best_D * 10000 + int(best_lr * 10000) * 10 + best_epochs
    rng = np.random.default_rng(seed)
    emb, W, b = train_model(X, y, C, best_D, best_lr, best_epochs, rng)

    # Write weights.bin
    # Format: magic (uint32), d (int32), D (int32), C (int32)
    # Then embeddings: d * 256 * D floats (float32) in order [pos][byte][dim]
    # Then W: C * d * D floats in order [class][pos][dim]
    # Then b: C floats
    magic = 0x454D424C  # "EMBL"
    d = 16
    out = struct.pack("<4i", magic, d, best_D, C)
    out += best_emb.astype(np.float32).tobytes()
    out += best_W.astype(np.float32).tobytes()
    out += best_b.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1] << 8 | (uint32_t)buf[*pos+2] << 16 | (uint32_t)buf[*pos+3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t len) {
    unsigned char hdr[4] = { len & 0xFF, (len >> 8) & 0xFF, (len >> 16) & 0xFF, (len >> 24) & 0xFF };
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, len, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long wsize, isize = 0;
    long N = atol(argv[3]);
    unsigned char *wbuf = slurp(argv[1], &wsize);
    unsigned char *ibuf = N ? slurp(argv[2], &isize) : NULL;

    // Parse header
    if (wsize < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wbuf;
    uint32_t magic = hdr[0];
    int d = hdr[1];
    int D = hdr[2];
    int C = hdr[3];
    if (magic != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    if (d != 16) { fprintf(stderr, "expected d=16, got %d\n", d); return 3; }
    if (C < 1 || C > 256) { fprintf(stderr, "bad C=%d\n", C); return 3; }
    if (D < 1 || D > 64) { fprintf(stderr, "bad D=%d\n", D); return 3; }

    // Expected size: 16 + 4*(d*256*D + C*d*D + C)
    long expected = 16 + 4L * ((long)d * 256 * D + (long)C * d * D + C);
    if (wsize != expected) { fprintf(stderr, "weights size mismatch: %ld != %ld\n", wsize, expected); return 3; }

    const float *emb = (const float *)(wbuf + 16);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * d * D;

    long ipos = 0;
    for (long i = 0; i < N; i++) {
        uint32_t reclen;
        const unsigned char *rec = next_rec(ibuf, isize, &ipos, &reclen);
        if (reclen != (uint32_t)d) { fprintf(stderr, "record %ld length %u != %d\n", i, reclen, d); return 3; }

        // Compute logits for each class
        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = b[c];
            const float *Wc = W + (long)c * d * D;
            for (int p = 0; p < d; p++) {
                unsigned char byte = rec[p];
                const float *e = emb + ((long)p * 256 + byte) * D;
                const float *w = Wc + (long)p * D;
                for (int k = 0; k < D; k++) {
                    score += w[k] * e[k];
                }
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```