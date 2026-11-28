```hypothesis
Family: label-fm (Factorization Machine with per-class pairwise interactions).
Why: The task is character-level language modeling with 16-byte context. Linear models miss pairwise interactions between bytes (e.g., "th" -> "e"). A factorization machine captures all pairwise interactions via low-rank embeddings per class, adding only O(d*C*k) parameters. With k=2..4, this should significantly improve accuracy over label-linear while keeping inference fast (O(d*C*k) operations). The model is deterministic and trains quickly with Adam.
Failure indication: If dev accuracy <= 0.3375 (label-linear) or inference cost > 2x label-linear, the model does not meet goals.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1
# Grid search hyperparameters
K_CHOICES = [2, 4]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
MAX_EPOCHS_TUNE = 5
MAX_EPOCHS_FULL = 30
SAMPLES_PER_EPOCH = 600000  # not used directly, we use epochs

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, k, rng):
    # bias (C), linear (d,256,C), emb (d,256,C,k)
    scale = 0.01
    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, scale, (d, 256, C))).astype(np.float32)
    emb = (rng.normal(0, scale, (d, 256, C, k))).astype(np.float32)
    return bias, linear, emb

def forward(X, bias, linear, emb):
    # X: (B, d) uint8
    B, d = X.shape
    C = bias.shape[0]
    k = emb.shape[3]
    # linear part
    logits = np.broadcast_to(bias, (B, C)).copy()
    for j in range(d):
        logits += linear[j, X[:, j], :]  # (B, C)
    # interaction part
    sum_emb = np.zeros((B, C, k), np.float32)
    sum_sq = np.zeros((B, C, k), np.float32)
    for j in range(d):
        e = emb[j, X[:, j], :, :]  # (B, C, k)
        sum_emb += e
        sum_sq += e * e
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq, axis=2)  # (B, C)
    logits += inter
    return logits, sum_emb, sum_sq

def loss_and_grad(X, y, bias, linear, emb, l2):
    B, d = X.shape
    C = bias.shape[0]
    k = emb.shape[3]
    logits, sum_emb, sum_sq = forward(X, bias, linear, emb)
    # softmax + cross entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # gradient of loss wrt logits
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    # loss
    loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
    # L2 regularization
    loss += 0.5 * l2 * (np.sum(linear * linear) + np.sum(emb * emb))
    # gradients
    dbias = dlogits.sum(axis=0)
    dlinear = np.zeros_like(linear)
    demb = np.zeros_like(emb)
    for j in range(d):
        xj = X[:, j]
        # linear gradient
        for b in range(B):
            dlinear[j, xj[b], :] += dlogits[b, :]
        # embedding gradient
        # dinter/demb = dlogits * (sum_emb - emb)
        e = emb[j, xj, :, :]  # (B, C, k)
        # sum_emb is (B, C, k)
        factor = dlogits[:, :, np.newaxis] * (sum_emb - e)  # (B, C, k)
        for b in range(B):
            demb[j, xj[b], :, :] += factor[b]
    # add L2
    dlinear += l2 * linear
    demb += l2 * emb
    return loss, dbias, dlinear, demb

def adam_update(params, grads, states, t, lr):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def accuracy(X, y, bias, linear, emb):
    logits, _, _ = forward(X, bias, linear, emb)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X, y, d, C, k, lr, l2, epochs, rng, verbose=False):
    bias, linear, emb = init_params(d, C, k, rng)
    params = [bias, linear, emb]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            loss, dbias, dlinear, demb = loss_and_grad(Xb, yb, bias, linear, emb, l2)
            epoch_loss += loss
            batches += 1
            t += 1
            adam_update(params, [dbias, dlinear, demb], states, t, lr)
        if verbose:
            print(f"epoch {epoch+1}: loss={epoch_loss/batches:.4f}")
    return bias, linear, emb

def evaluate_tuning(X_train, y_train, X_val, y_val, d, C):
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_hp = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # train on training split
                bias, linear, emb = train_model(X_train, y_train, d, C, k, lr, l2, MAX_EPOCHS_TUNE, rng)
                acc = accuracy(X_val, y_val, bias, linear, emb)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (bias, linear, emb)
                    best_hp = (k, lr, l2)
    return best_params, best_hp, best_acc

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split last 20% for tuning
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}", file=sys.stderr)
    # hyperparameter tuning
    best_params, best_hp, best_val_acc = evaluate_tuning(X_train, y_train, X_val, y_val, d, C)
    k, lr, l2 = best_hp
    print(f"Best hp: k={k}, lr={lr}, l2={l2}, val_acc={best_val_acc:.4f}", file=sys.stderr)
    # retrain on full data
    rng = np.random.default_rng(SEED)
    bias, linear, emb = train_model(X, y, d, C, k, lr, l2, MAX_EPOCHS_FULL, rng, verbose=True)
    # save weights
    # magic 0x464D0001, d, C, k
    magic = 0x464D0001
    header = struct.pack("<Iiii", magic, d, C, k)
    # flatten arrays in C order: bias (C), linear (d,256,C), emb (d,256,C,k)
    out = header
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)
    print(f"Saved weights: {len(out)} bytes", file=sys.stderr)

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D0001) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], C = hd[1], k = hd[2];
    if (d != 16 || C != 65 || k < 1 || k > 8) { fprintf(stderr, "bad weights: invalid dims\n"); return 3; }
    // expected floats: bias(C) + linear(d*256*C) + emb(d*256*C*k)
    long expected_floats = C + (long)d * 256 * C + (long)d * 256 * C * k;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *fptr = (float *)(wb + 16);
    float *bias = fptr; fptr += C;
    float *linear = fptr; fptr += (long)d * 256 * C;
    float *emb = fptr;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    // allocate temporary arrays for sum_emb and sum_sq_emb
    float *sum_emb = malloc(sizeof(float) * C * k);
    float *sum_sq = malloc(sizeof(float) * C * k);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        // initialize logits with bias
        memcpy(logits, bias, C * sizeof(float));
        // zero accumulators
        memset(sum_emb, 0, C * k * sizeof(float));
        memset(sum_sq, 0, C * k * sizeof(float));
        // loop over positions
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            // linear part
            const float *lin_ptr = linear + ((long)j * 256 + byte) * C;
            for (int c = 0; c < C; c++) logits[c] += lin_ptr[c];
            // embedding part
            const float *emb_ptr = emb + (((long)j * 256 + byte) * C * k);
            for (int c = 0; c < C; c++) {
                const float *e = emb_ptr + c * k;
                float *se = sum_emb + c * k;
                float *sq = sum_sq + c * k;
                for (int f = 0; f < k; f++) {
                    float val = e[f];
                    se[f] += val;
                    sq[f] += val * val;
                }
            }
        }
        // add interaction term
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            float *se = sum_emb + c * k;
            float *sq = sum_sq + c * k;
            for (int f = 0; f < k; f++) {
                inter += se[f] * se[f] - sq[f];
            }
            logits[c] += 0.5f * inter;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```