```hypothesis
Family: label-fm (Factorization Machine with per-position embeddings).
Why: The task predicts the next character from 16 bytes of context. A linear model (label-linear) only uses independent per-position features. An FM adds all pairwise interactions via low-rank embeddings, capturing n-gram statistics crucial for text. The model is tiny (~5 MB for k=4) and inference is O(C*d*k) ~ 4k operations/item, far cheaper than label-gbdt. It should beat label-linear's 0.3375 accuracy while staying well below its 665 ns/item.
Failure: If dev accuracy ≤ 0.3375 or CPU cost ≥ 665 ns/item, the family does not meet the goal.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter search space (tuned on the last 20% of train.in)
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.1, 0.03, 0.01, 0.003]
SEARCH_EPOCHS = 10
FINAL_EPOCHS = 30
BATCH = 64
L2 = 1e-4
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, k, rng):
    # linear: (d, 256, C)
    linear = rng.normal(0, 0.01, size=(d, 256, C)).astype(np.float32)
    # embeddings: (d, 256, C, k)
    embeddings = rng.normal(0, 0.01, size=(d, 256, C, k)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    return linear, embeddings, bias

def forward(indices, linear, embeddings, bias):
    # indices: (B, d) uint8
    B, d = indices.shape
    C = linear.shape[2]
    k = embeddings.shape[3]
    # Linear term
    scores = bias.reshape(1, C) + np.zeros((B, C), dtype=np.float32)
    for j in range(d):
        idx = indices[:, j]  # (B,)
        scores += linear[j, idx]  # (B, C)
    # FM term
    sum_vec = np.zeros((B, C, k), dtype=np.float32)
    sum_sq_vec = np.zeros((B, C, k), dtype=np.float32)
    for j in range(d):
        idx = indices[:, j]
        emb = embeddings[j, idx]  # (B, C, k)
        sum_vec += emb
        sum_sq_vec += emb * emb
    fm = 0.5 * np.sum(sum_vec * sum_vec - sum_sq_vec, axis=2)  # (B, C)
    scores += fm
    return scores

def loss_and_grad(indices, labels, linear, embeddings, bias, L2):
    B, d = indices.shape
    C = linear.shape[2]
    k = embeddings.shape[3]
    # Forward
    scores = forward(indices, linear, embeddings, bias)
    # Softmax cross-entropy
    max_s = np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    log_probs = np.log(probs + 1e-12)
    loss = -np.mean(log_probs[np.arange(B), labels])
    # Gradient of loss w.r.t scores
    grad_scores = probs.copy()
    grad_scores[np.arange(B), labels] -= 1.0
    grad_scores /= B  # (B, C)
    # Gradients
    grad_linear = np.zeros_like(linear)
    grad_embeddings = np.zeros_like(embeddings)
    grad_bias = np.zeros_like(bias)
    # Bias gradient
    grad_bias = np.sum(grad_scores, axis=0)
    # Linear gradient
    for j in range(d):
        idx = indices[:, j]  # (B,)
        # For each example, add grad_scores to grad_linear[j, idx]
        np.add.at(grad_linear[j], idx, grad_scores)
    # FM gradient
    # For each class c, we need sum of embeddings over positions: sum_vec (B, C, k)
    sum_vec = np.zeros((B, C, k), dtype=np.float32)
    for j in range(d):
        idx = indices[:, j]
        sum_vec += embeddings[j, idx]
    # grad_emb for each position j: grad_scores * (sum_vec - emb_j)
    for j in range(d):
        idx = indices[:, j]
        emb_j = embeddings[j, idx]  # (B, C, k)
        # grad = grad_scores[:,:,None] * (sum_vec - emb_j)
        grad = grad_scores[:, :, None] * (sum_vec - emb_j)
        np.add.at(grad_embeddings[j], idx, grad)
    # L2 regularization
    grad_linear += L2 * linear
    grad_embeddings += L2 * embeddings
    loss += 0.5 * L2 * (np.sum(linear * linear) + np.sum(embeddings * embeddings))
    return loss, grad_linear, grad_embeddings, grad_bias

def adam_step(params, grads, states, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def evaluate(indices, labels, linear, embeddings, bias):
    scores = forward(indices, linear, embeddings, bias)
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == labels)

def train_loop(X, y, linear, embeddings, bias, epochs, lr, rng):
    n = X.shape[0]
    params = [linear, embeddings, bias]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            loss, gl, ge, gb = loss_and_grad(xi, yi, linear, embeddings, bias, L2)
            t += 1
            adam_step(params, [gl, ge, gb], states, lr, t)
    return linear, embeddings, bias

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Read all data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_acc = -1.0
    best_k = None
    best_lr = None
    # Hyperparameter search
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            rng = np.random.default_rng(SEED + k * 100 + int(lr * 10000))
            linear, embeddings, bias = init_params(d, C, k, rng)
            linear, embeddings, bias = train_loop(X_train, y_train, linear, embeddings, bias,
                                                 SEARCH_EPOCHS, lr, rng)
            acc = evaluate(X_val, y_val, linear, embeddings, bias)
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_lr = lr
    # Final training on all data
    rng = np.random.default_rng(SEED)
    linear, embeddings, bias = init_params(d, C, best_k, rng)
    linear, embeddings, bias = train_loop(X, y, linear, embeddings, bias,
                                         FINAL_EPOCHS, best_lr, rng)
    # Write weights.bin
    # Header: magic, d, k, C
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, best_k, C)
    # Linear: (d, 256, C) -> flat
    linear_bytes = linear.astype("<f4").tobytes()
    # Bias: (C,)
    bias_bytes = bias.astype("<f4").tobytes()
    # Embeddings: (d, 256, C, k) -> flat
    emb_bytes = embeddings.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + linear_bytes + bias_bytes + emb_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * C * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *bias = linear + (long)d * 256 * C;
    const float *emb = bias + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_vec = malloc(sizeof(float) * C * k);
    float *sum_sq_vec = malloc(sizeof(float) * C * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        // Linear + bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            long base = ((long)j * 256 + byte) * C;
            for (int c = 0; c < C; c++) scores[c] += linear[base + c];
        }
        // FM term
        for (int c = 0; c < C; c++) {
            for (int u = 0; u < k; u++) {
                sum_vec[c * k + u] = 0.0f;
                sum_sq_vec[c * k + u] = 0.0f;
            }
        }
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            long base = (((long)j * 256 + byte) * C + 0) * k;
            for (int c = 0; c < C; c++) {
                for (int u = 0; u < k; u++) {
                    float val = emb[base + c * k + u];
                    sum_vec[c * k + u] += val;
                    sum_sq_vec[c * k + u] += val * val;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float fm = 0.0f;
            for (int u = 0; u < k; u++) {
                float sv = sum_vec[c * k + u];
                fm += 0.5f * (sv * sv - sum_sq_vec[c * k + u]);
            }
            scores[c] += fm;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```