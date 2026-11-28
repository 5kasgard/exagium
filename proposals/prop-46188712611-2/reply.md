```hypothesis
Family: label-embmlp (embedding + one hidden layer MLP). Each of the 16 context positions gets a learned character embedding (256 x D) and a positional embedding (16 x D). They are summed per position, then summed across positions to a D-vector. A ReLU hidden layer of size H projects to 65 output classes. This captures position-sensitive n-gram interactions with far fewer parameters than one-hot linear (D=16, H=32 gives ~30k params vs 266k). Should beat label-linear (0.3375) on dev accuracy while keeping inference cost low (dense D-vector + small MLP). Failure: if dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2x.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen after tuning on the last 20% of train.in
SEED = 1
D = 16          # embedding dimension
H = 32          # hidden layer size
LR = 1e-3       # learning rate
L2 = 1e-5       # weight decay
BATCH = 128     # minibatch size
EPOCHS_SEARCH = 10   # epochs for hyperparameter search
EPOCHS_FINAL = 30    # epochs for final training on all data
VAL_FRAC = 0.2       # last 20% for validation


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(D, H, C, rng):
    # char_emb: 256 x D, pos_emb: 16 x D, W1: D x H, b1: H, W2: H x C, b2: C
    scale = math.sqrt(6.0 / (256 + D))
    char_emb = rng.uniform(-scale, scale, (256, D)).astype(np.float32)
    scale = math.sqrt(6.0 / (16 + D))
    pos_emb = rng.uniform(-scale, scale, (16, D)).astype(np.float32)
    scale = math.sqrt(6.0 / (D + H))
    W1 = rng.uniform(-scale, scale, (D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    scale = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale, scale, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [char_emb, pos_emb, W1, b1, W2, b2]


def forward(params, X_batch):
    char_emb, pos_emb, W1, b1, W2, b2 = params
    # X_batch: (B, 16) uint8
    B = X_batch.shape[0]
    # gather character embeddings: (B, 16, D)
    x_emb = char_emb[X_batch]  # fancy indexing
    # add positional embeddings (broadcast)
    x_emb += pos_emb[np.newaxis, :, :]  # (1, 16, D)
    # sum over positions
    summed = x_emb.sum(axis=1)  # (B, D)
    # hidden layer
    h = np.maximum(summed @ W1 + b1, 0, dtype=np.float32)  # (B, H)
    logits = h @ W2 + b2  # (B, C)
    return logits, h, summed


def loss_and_grad(params, X_batch, y_batch):
    char_emb, pos_emb, W1, b1, W2, b2 = params
    B = X_batch.shape[0]
    C = b2.shape[0]

    logits, h, summed = forward(params, X_batch)

    # softmax + cross-entropy gradient
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (B, C)
    probs[np.arange(B), y_batch] -= 1.0
    probs /= B  # dL/dlogits

    # gradients for output layer
    grad_W2 = h.T @ probs  # (H, C)
    grad_b2 = probs.sum(axis=0)  # (C,)

    # backprop to hidden layer
    grad_h = probs @ W2.T  # (B, H)
    grad_h[h == 0] = 0.0  # ReLU derivative

    # gradients for hidden layer
    grad_W1 = summed.T @ grad_h  # (D, H)
    grad_b1 = grad_h.sum(axis=0)  # (H,)

    # backprop to summed embeddings
    grad_summed = grad_h @ W1.T  # (B, D)

    # gradients for positional embeddings (same for all positions in a batch)
    grad_pos_emb = np.zeros_like(pos_emb)
    for p in range(16):
        grad_pos_emb[p] = grad_summed.sum(axis=0)

    # gradients for character embeddings (scatter add)
    grad_char_emb = np.zeros_like(char_emb)
    for p in range(16):
        bytes_p = X_batch[:, p]
        np.add.at(grad_char_emb, bytes_p, grad_summed)

    # L2 regularization gradients
    l2_params = [char_emb, pos_emb, W1, W2]
    l2_grads = [L2 * p for p in l2_params]
    grad_char_emb += l2_grads[0]
    grad_pos_emb += l2_grads[1]
    grad_W1 += l2_grads[2]
    grad_W2 += l2_grads[3]

    return (grad_char_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2), logits


def evaluate(params, X, y):
    logits, _, _ = forward(params, X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_loop(params, X_train, y_train, X_val, y_val, epochs, lr, batch_size, rng):
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = X_train.shape[0]
    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            grads, _ = loss_and_grad(params, X_batch, y_batch)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # validation accuracy
        val_acc = evaluate(params, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    return best_params, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # should be 16
    assert d == 16

    n = X.shape[0]
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val
    # last 20% in file order is validation
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    rng = np.random.default_rng(SEED)

    # Hyperparameter grid (D, H, lr, epochs_search)
    # We hardcode the best found after offline tuning; here we just use one config.
    configs = [
        (D, H, LR, EPOCHS_SEARCH),
    ]

    best_overall_acc = -1.0
    best_overall_params = None

    for D_cfg, H_cfg, lr_cfg, epochs_cfg in configs:
        print(f"Trying D={D_cfg}, H={H_cfg}, lr={lr_cfg}", file=sys.stderr)
        params = init_params(D_cfg, H_cfg, C, rng)
        best_params, val_acc = train_loop(params, X_train, y_train, X_val, y_val, epochs_cfg, lr_cfg, BATCH, rng)
        print(f"  Val accuracy: {val_acc:.4f}", file=sys.stderr)
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_params = best_params

    # Retrain on all data with best config
    print(f"Best val accuracy: {best_overall_acc:.4f}, retraining on all data...", file=sys.stderr)
    # Re-initialize with best config (using same D, H)
    params = init_params(D, H, C, rng)
    # We could continue from best_overall_params, but to be safe we retrain from scratch on all data.
    # Use the best config's hyperparameters.
    all_params, _ = train_loop(params, X, y, X_val, y_val, EPOCHS_FINAL, LR, BATCH, rng)

    # Save weights
    char_emb, pos_emb, W1, b1, W2, b2 = all_params
    # Magic: 0x4C424D45 ('EMBL' little-endian)
    magic = 0x4C424D45
    header = struct.pack("<5i", magic, D, H, C, d)
    # Order: char_emb (256*D), pos_emb (16*D), W1 (D*H), b1 (H), W2 (H*C), b2 (C)
    body = (
        char_emb.astype("<f4").tobytes() +
        pos_emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int D = hd[1], H = hd[2], C = hd[3], d = hd[4];
    if (d != 16 || C < 1 || C > 256 || D < 1 || H < 1) {
        fprintf(stderr, "bad weights dimensions\n"); return 3;
    }
    // Expected size: header(20) + char_emb(256*D*4) + pos_emb(16*D*4) + W1(D*H*4) + b1(H*4) + W2(H*C*4) + b2(C*4)
    long expected = 20 +
        256L * D * 4 +
        16L * D * 4 +
        (long)D * H * 4 +
        (long)H * 4 +
        (long)H * C * 4 +
        (long)C * 4;
    if (nw != expected) {
        fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected);
        return 3;
    }

    const float *p = (const float *)(hd + 5); // after 5 int32s = 20 bytes
    const float *char_emb = p; p += 256 * D;
    const float *pos_emb = p; p += 16 * D;
    const float *W1 = p; p += D * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = (float *)malloc(D * sizeof(float));
    float *hidden = (float *)malloc(H * sizeof(float));
    float *logits = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // summed = sum_p (char_emb[xi[p]] + pos_emb[p])
        memset(summed, 0, D * sizeof(float));
        for (int p_idx = 0; p_idx < d; p_idx++) {
            const float *cemb = char_emb + xi[p_idx] * D;
            const float *pemb = pos_emb + p_idx * D;
            for (int j = 0; j < D; j++) summed[j] += cemb[j] + pemb[j];
        }

        // hidden = ReLU(summed @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int j = 0; j < D; j++) acc += summed[j] * W1[j * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[h * C + c];
            logits[c] = acc;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(summed); free(hidden); free(logits);
    return 0;
}
```