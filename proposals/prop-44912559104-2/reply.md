```hypothesis
Family: label-embed (factorized linear with per-position byte embeddings).
Why: The input is 16 categorical bytes; label-linear uses a full one-hot weight matrix (16*256*65 params). By learning a shared embedding of dimension E per byte value per position and summing them, we reduce parameters (16*256*E + E*65) and encourage generalization across positions. For small E (e.g., 32) this should beat label-linear's 0.3375 accuracy at similar or lower inference cost (embedding lookup + small matvec). Failure: dev accuracy ≤ label-linear or inference cost (ns/item) significantly higher.
```

```python
"""label-embed: factorized linear model with per-position byte embeddings.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Chooses embedding size E on last 20% of train.in, then retrains on all data.
Weights format: magic 0x454D4244, int32 d, E, C; float32 embeddings[d*256*E], W_out[E*C], b_out[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (same as label-linear)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_E = [16, 32, 64, 128]  # embedding sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t. logits."""
    # logits: (batch, C), y: (batch,)
    batch = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return (probs / batch).astype(np.float32)


def evaluate(X, y, emb, W_out, b_out):
    """Compute accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        # embedded = sum_j emb[j, Xb[:,j], :]
        embedded = np.zeros((Xb.shape[0], emb.shape[2]), dtype=np.float32)
        for j in range(Xb.shape[1]):
            embedded += emb[j, Xb[:, j]]
        logits = embedded @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X, y, E, rng, epochs, verbose=False):
    """Train embeddings and output layer for given E, return (emb, W_out, b_out)."""
    n, d = X.shape
    C = W_out.shape[1] if 'W_out' in locals() else 65  # placeholder
    C = int(np.max(y)) + 1  # but we know C from card
    # Initialize parameters
    emb = rng.uniform(-1, 1, (d, 256, E)).astype(np.float32) * math.sqrt(6.0 / 256)
    W_out = rng.uniform(-1, 1, (E, C)).astype(np.float32) * math.sqrt(6.0 / E)
    b_out = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            batch = Xb.shape[0]

            # Forward
            embedded = np.zeros((batch, E), dtype=np.float32)
            for j in range(d):
                embedded += emb[j, Xb[:, j]]
            logits = embedded @ W_out + b_out

            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)

            # Gradients
            dW_out = embedded.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(axis=0)
            dembedded = dlogits @ W_out.T  # (batch, E)

            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], Xb[:, j], dembedded)

            # Adam updates
            t += 1
            for param, grad, m, v in [
                (emb, demb, m_emb, v_emb),
                (W_out, dW_out, m_W, v_W),
                (b_out, db_out, m_b, v_b),
            ]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return emb, W_out, b_out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for tuning
    n_tune = n // 5
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    best_E, best_acc, best_params = None, -1.0, None

    for E in CANDIDATE_E:
        rng = np.random.default_rng(SEED + E * 1000)
        emb, W_out, b_out = train_model(X_train, y_train, E, rng, epochs)
        acc = evaluate(X_tune, y_tune, emb, W_out, b_out)
        if acc > best_acc:
            best_acc, best_E = acc, E
            best_params = (emb.copy(), W_out.copy(), b_out.copy())

    # Retrain on full data with best_E
    rng = np.random.default_rng(SEED + best_E * 1000)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W_out, b_out = train_model(X, y, best_E, rng, epochs_full)

    # Write weights.bin
    # magic, d, E, C
    header = struct.pack("<4i", 0x454D4244, d, best_E, C)
    # embeddings: d * 256 * E floats (order: position, byte, embedding)
    emb_flat = emb.astype("<f4").tobytes()
    W_out_flat = W_out.astype("<f4").tobytes()
    b_out_flat = b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_flat + W_out_flat + b_out_flat)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index.
 * Weights format: magic 0x454D4244, int32 d, E, C; float32 emb[d*256*E], W_out[E*C], b_out[C]. */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(sum_emb, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) sum_emb[e_idx] += e[e_idx];
        }

        memcpy(scores, b_out, sizeof(float) * C);
        for (int e_idx = 0; e_idx < E; e_idx++) {
            float v = sum_emb[e_idx];
            if (v == 0.0f) continue;
            const float *w = W_out + (long)e_idx * C;
            for (int c = 0; c < C; c++) scores[c] += v * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```