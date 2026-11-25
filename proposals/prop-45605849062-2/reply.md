```hypothesis
We propose a factorization machine (label-fm) with class-specific embeddings. It captures pairwise interactions between the 16 context bytes via low-rank embeddings (K=2), which is crucial for character-level language modeling where local character interactions matter. The model uses a sum-v trick for O(d*K) interaction evaluation, keeping inference fast. If it fails to beat label-linear, it would show that pairwise interactions are not sufficient or the embedding rank is too small for this task.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
K = 2                     # embedding rank
LR_GRID = [0.1, 0.03, 0.01]
L2_GRID = [1e-4, 1e-3]
BATCH = 64
TUNE_EPOCHS = 15
FULL_EPOCHS = 40
SEED = 1
MAGIC = 0x464D4C31       # "FML1"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class FMModel:
    def __init__(self, d, C, K, seed):
        self.d, self.C, self.K = d, C, K
        self.rng = np.random.default_rng(seed)
        # parameters
        self.bias = np.zeros(C, dtype=np.float32)
        scale = math.sqrt(6.0 / (256 * d))
        self.W = (self.rng.uniform(-1, 1, (d, 256, C)) * scale).astype(np.float32)
        self.V = (self.rng.uniform(-1, 1, (d, 256, C, K)) * scale).astype(np.float32)
        # Adam state
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_V = np.zeros_like(self.V)
        self.v_V = np.zeros_like(self.V)
        self.t = 0

    def forward(self, X):
        n = X.shape[0]
        # linear part
        linear = self.bias + np.sum(self.W[np.arange(self.d), X.T, :], axis=0).T
        # interaction part
        inter = np.zeros((n, self.C), dtype=np.float32)
        for c in range(self.C):
            Vc = self.V[:, :, c, :]          # (d, 256, K)
            sum_v = np.zeros((n, self.K), dtype=np.float32)
            sum_sq = np.zeros((n, self.K), dtype=np.float32)
            for i in range(self.d):
                v = Vc[i, X[:, i], :]        # (n, K)
                sum_v += v
                sum_sq += v * v
            inter[:, c] = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
        return linear + inter

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def _adam_step(self, param, grad, m, v, lr, l2, eps=1e-8):
        grad = grad + l2 * param
        self.t += 1
        m *= 0.9; m += 0.1 * grad
        v *= 0.999; v += 0.001 * grad * grad
        m_hat = m / (1 - 0.9 ** self.t)
        v_hat = v / (1 - 0.999 ** self.t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_batch(self, Xb, yb, lr, l2):
        b = Xb.shape[0]
        # forward
        scores = self.forward(Xb)                     # (b, C)
        # softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores, dtype=np.float32)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        # gradient of cross-entropy
        probs[np.arange(b), yb] -= 1.0
        grad_scores = probs / b                       # (b, C)

        # grad_bias
        grad_bias = grad_scores.sum(axis=0)
        # grad_W
        grad_W = np.zeros_like(self.W)
        for i in range(self.d):
            # for each byte value, accumulate grad_scores where Xb[:,i]==v
            for v in range(256):
                mask = (Xb[:, i] == v)
                if mask.any():
                    grad_W[i, v, :] = grad_scores[mask].sum(axis=0)
        # grad_V
        grad_V = np.zeros_like(self.V)
        for c in range(self.C):
            Vc = self.V[:, :, c, :]
            grad_inter = grad_scores[:, c]            # (b,)
            # compute sum_v per example for this class
            sum_v = np.zeros((b, self.K), dtype=np.float32)
            for i in range(self.d):
                sum_v += Vc[i, Xb[:, i], :]
            for i in range(self.d):
                v_i = Vc[i, Xb[:, i], :]              # (b, K)
                # grad_v_i = grad_inter[:,None] * (sum_v - v_i)
                g = grad_inter[:, None] * (sum_v - v_i)
                # accumulate into grad_V
                for v in range(256):
                    mask = (Xb[:, i] == v)
                    if mask.any():
                        grad_V[i, v, c, :] = g[mask].sum(axis=0)

        # Adam updates
        self._adam_step(self.bias, grad_bias, self.m_bias, self.v_bias, lr, l2)
        self._adam_step(self.W, grad_W, self.m_W, self.v_W, lr, l2)
        self._adam_step(self.V, grad_V, self.m_V, self.v_V, lr, l2)

    def train(self, X, y, lr, l2, batch, epochs, verbose=False):
        n = X.shape[0]
        for ep in range(epochs):
            order = self.rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s+batch]
                self.train_batch(X[idx], y[idx], lr, l2)
            if verbose:
                preds = self.predict(X)
                acc = (preds == y).mean()
                print(f"epoch {ep+1}: train acc={acc:.4f}")

    def save(self, path):
        # pack: magic, d, C, K, bias, W, V
        header = struct.pack("<4i", MAGIC, self.d, self.C, self.K)
        bias_b = self.bias.astype("<f4").tobytes()
        W_b = self.W.astype("<f4").tobytes()
        V_b = self.V.astype("<f4").tobytes()
        with open(path, "wb") as f:
            f.write(header + bias_b + W_b + V_b)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_lr, best_l2 = LR_GRID[0], L2_GRID[0]
    for lr in LR_GRID:
        for l2 in L2_GRID:
            model = FMModel(d, C, K, SEED)
            model.train(X_tr, y_tr, lr, l2, BATCH, TUNE_EPOCHS, verbose=False)
            preds = model.predict(X_tune)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc, best_lr, best_l2 = acc, lr, l2
    print(f"tune best acc={best_acc:.4f} lr={best_lr} l2={best_l2}")

    model = FMModel(d, C, K, SEED)
    model.train(X, y, best_lr, best_l2, BATCH, FULL_EPOCHS, verbose=True)
    model.save(sys.argv[4])

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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 16 || C != 65 || K != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L
        + 4L * C
        + 4L * d * 256 * C
        + 4L * d * 256 * C * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    const float *V = W + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[2], sum_sq[2];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        unsigned char best = 0;
        float best_score = -1e30f;
        for (int c = 0; c < C; c++) {
            float score = bias[c];
            // linear part
            for (int j = 0; j < d; j++) {
                score += W[((long)j * 256 + xi[j]) * C + c];
            }
            // interaction part using sum_v trick
            sum_v[0] = sum_v[1] = 0.0f;
            sum_sq[0] = sum_sq[1] = 0.0f;
            for (int j = 0; j < d; j++) {
                const float *v = V + ((((long)j * 256 + xi[j]) * C + c) * K);
                sum_v[0] += v[0]; sum_v[1] += v[1];
                sum_sq[0] += v[0] * v[0]; sum_sq[1] += v[1] * v[1];
            }
            float inter = 0.5f * ((sum_v[0]*sum_v[0] - sum_sq[0]) + (sum_v[1]*sum_v[1] - sum_sq[1]));
            score += inter;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```