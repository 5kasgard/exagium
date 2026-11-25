```hypothesis
We propose a position-wise embedding model (label-posemb) that learns a dense embedding for each byte at each of the 16 context positions, followed by a position-specific linear projection to 65 classes. This captures positional n-gram statistics with far fewer parameters than a full one-hot bigram model, and should outperform label-linear (0.3375) by modeling position-specific character distributions. If dev accuracy does not exceed 0.34, the approach fails.
```

```python
"""Position-wise embedding model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Format: int32 magic 0x50454D42, d, E, C; float32 emb[d][256][E], Wout[d][E][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E_CANDIDATES = [4, 8, 16, 32]  # embedding dimensions to try on tuning split
TUNE_EPOCHS = 30  # epochs for hyperparameter selection


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_size = int(n * tune_frac)
    return X[:-tune_size], y[:-tune_size], X[-tune_size:], y[-tune_size:]


def init_params(d, E, C, rng):
    # emb: (d, 256, E)
    emb = rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    # Wout: (d, E, C)
    Wout = rng.normal(0, math.sqrt(6.0 / (E + C)), size=(d, E, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    return emb, Wout, bias


def forward_logits(X, emb, Wout, bias):
    # X: (n, d) uint8
    n, d = X.shape
    logits = np.broadcast_to(bias, (n, bias.shape[0])).copy()
    for j in range(d):
        # gather embeddings for position j: (n, E)
        e = emb[j, X[:, j], :]
        logits += e @ Wout[j]
    return logits


def loss_and_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    grad_logits = probs / n
    loss = -np.log(np.maximum(probs[np.arange(n), y] + 1.0, 1e-12)).mean()  # actually cross-entropy
    # correct loss:
    log_probs = logits - np.log(exp_logits.sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()
    return loss, grad_logits.astype(np.float32)


def backward(grad_logits, X, emb, Wout):
    n, d = X.shape
    _, E, C = Wout.shape
    grad_emb = np.zeros_like(emb)
    grad_Wout = np.zeros_like(Wout)
    grad_bias = grad_logits.sum(axis=0)
    for j in range(d):
        e = emb[j, X[:, j], :]  # (n, E)
        grad_Wout[j] = e.T @ grad_logits + L2 * Wout[j]
        g_e = grad_logits @ Wout[j].T  # (n, E)
        # scatter-add to grad_emb
        for i in range(n):
            grad_emb[j, X[i, j], :] += g_e[i]
    grad_emb += L2 * emb
    return grad_emb, grad_Wout, grad_bias


def adam_update(params, grads, states, t, lr=LR):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def accuracy(X, y, emb, Wout, bias):
    logits = forward_logits(X, emb, Wout, bias)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_model(X_tr, y_tr, X_val, y_val, d, E, C, rng, epochs):
    emb, Wout, bias = init_params(d, E, C, rng)
    params = [emb, Wout, bias]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X_tr.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits = forward_logits(Xb, emb, Wout, bias)
            _, grad_logits = loss_and_grad(logits, yb)
            grad_emb, grad_Wout, grad_bias = backward(grad_logits, Xb, emb, Wout)
            t += 1
            adam_update(params, [grad_emb, grad_Wout, grad_bias], states, t)
    val_acc = accuracy(X_val, y_val, emb, Wout, bias)
    return val_acc, emb, Wout, bias


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    rng = np.random.default_rng(SEED)

    # Split for tuning
    X_tr, y_tr, X_tune, y_tune = split_train_tune(X, y)

    # Hyperparameter search on tuning split
    best_E = E_CANDIDATES[0]
    best_acc = -1.0
    for E in E_CANDIDATES:
        # Use a fresh RNG stream for each candidate to keep determinism across candidates? 
        # We'll just use the same global rng; the state advances deterministically.
        val_acc, _, _, _ = train_model(X_tr, y_tr, X_tune, y_tune, d, E, C, rng, TUNE_EPOCHS)
        if val_acc > best_acc:
            best_acc = val_acc
            best_E = E

    # Retrain on full data with best_E
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))
    _, emb, Wout, bias = train_model(X, y, X, y, d, best_E, C, rng, epochs)  # X_val not used

    # Save weights
    magic = 0x50454D42  # 'PEMB' in little-endian byte order
    header = struct.pack("<4i", magic, d, best_E, C)
    # Ensure little-endian float32
    emb_bytes = emb.astype("<f4").tobytes()
    Wout_bytes = Wout.astype("<f4").tobytes()
    bias_bytes = bias.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header + emb_bytes + Wout_bytes + bias_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Position-wise embedding model inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x50454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 128) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + (long)d * 256 * E * 4 + (long)d * E * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *Wout = emb + (long)d * 256 * E;
    const float *bias = Wout + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *temp = malloc(sizeof(float) * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(logits, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            const float *W = Wout + ((long)j * E * C);
            // compute e @ W -> add to logits
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W[ee * C + c];
                logits[c] += sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(logits); free(temp);
    return 0;
}
```