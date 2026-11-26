```hypothesis
Family: label-fm (Factorization Machine with factorized pairwise interactions).
Why: Adult has mixed categorical/numeric features where interactions matter (e.g., education×occupation). FM captures pairwise interactions with O(d*k) inference via the sum_emb trick, staying fast. Linear models miss interactions; MLP/GBDT are heavier. A small k=4 FM should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at similar cost to linear.
Failure: If dev accuracy ≤0.8524 or instructions/item >1500, the model doesn't meet the goal.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning split selection
GRID = [
    {"k": 4, "lr": 0.01, "l2": 1e-4, "epochs": 10},
    {"k": 4, "lr": 0.03, "l2": 1e-4, "epochs": 10},
    {"k": 4, "lr": 0.01, "l2": 1e-3, "epochs": 10},
    {"k": 4, "lr": 0.03, "l2": 1e-3, "epochs": 10},
    {"k": 8, "lr": 0.01, "l2": 1e-4, "epochs": 10},
    {"k": 8, "lr": 0.03, "l2": 1e-4, "epochs": 10},
    {"k": 8, "lr": 0.01, "l2": 1e-3, "epochs": 10},
    {"k": 8, "lr": 0.03, "l2": 1e-3, "epochs": 10},
]
SEED = 1
BATCH = 64
FINAL_EPOCHS = 20  # for full retrain

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bce_logits(logits, y):
    # stable BCE with logits
    max_logits = np.maximum(logits, 0)
    log_exp = np.log(np.exp(-max_logits) + np.exp(logits - max_logits))
    return np.mean(max_logits - logits * y + log_exp)

class FMModel:
    def __init__(self, d, k, rng):
        self.d = d
        self.k = k
        self.bias = 0.0
        self.W = np.zeros((d, 256), dtype=np.float32)  # linear weights
        self.Emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)  # embeddings
        # Adam state
        self.m_bias, self.v_bias = 0.0, 0.0
        self.m_W, self.v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        self.m_Emb, self.v_Emb = np.zeros_like(self.Emb), np.zeros_like(self.Emb)
        self.t = 0

    def forward(self, X):
        # X: (n, d) uint8
        n = X.shape[0]
        lin = np.zeros(n, dtype=np.float32)
        sum_emb = np.zeros((n, self.k), dtype=np.float32)
        sum_emb_sq = np.zeros((n, self.k), dtype=np.float32)
        for j in range(self.d):
            lin += self.W[j, X[:, j]]
            emb_j = self.Emb[j, X[:, j], :]  # (n, k)
            sum_emb += emb_j
            sum_emb_sq += emb_j * emb_j
        inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
        return self.bias + lin + inter

    def backward(self, X, grad_logits):
        # grad_logits: (n,)
        n = X.shape[0]
        grad_bias = np.sum(grad_logits)
        grad_W = np.zeros_like(self.W)
        grad_Emb = np.zeros_like(self.Emb)
        # For linear part
        for j in range(self.d):
            # accumulate grad_logits per value
            vals = X[:, j]
            for v in range(256):
                mask = (vals == v)
                if np.any(mask):
                    grad_W[j, v] = np.sum(grad_logits[mask])
        # For embeddings
        # First compute sum_emb for the batch
        sum_emb = np.zeros((n, self.k), dtype=np.float32)
        for j in range(self.d):
            sum_emb += self.Emb[j, X[:, j], :]
        # For each feature, grad_emb = grad_logits[:,None] * (sum_emb - emb_j)
        for j in range(self.d):
            emb_j = self.Emb[j, X[:, j], :]  # (n, k)
            g = grad_logits[:, None] * (sum_emb - emb_j)  # (n, k)
            # scatter add to grad_Emb[j, X[:,j], :]
            for v in range(256):
                mask = (X[:, j] == v)
                if np.any(mask):
                    grad_Emb[j, v, :] = np.sum(g[mask], axis=0)
        return grad_bias, grad_W, grad_Emb

    def adam_step(self, grad_bias, grad_W, grad_Emb, lr, l2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        # bias
        g = grad_bias
        self.m_bias = beta1 * self.m_bias + (1 - beta1) * g
        self.v_bias = beta2 * self.v_bias + (1 - beta2) * g * g
        m_hat = self.m_bias / (1 - beta1 ** self.t)
        v_hat = self.v_bias / (1 - beta2 ** self.t)
        self.bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # W
        g = grad_W + l2 * self.W
        self.m_W = beta1 * self.m_W + (1 - beta1) * g
        self.v_W = beta2 * self.v_W + (1 - beta2) * g * g
        m_hat = self.m_W / (1 - beta1 ** self.t)
        v_hat = self.v_W / (1 - beta2 ** self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # Emb
        g = grad_Emb + l2 * self.Emb
        self.m_Emb = beta1 * self.m_Emb + (1 - beta1) * g
        self.v_Emb = beta2 * self.v_Emb + (1 - beta2) * g * g
        m_hat = self.m_Emb / (1 - beta1 ** self.t)
        v_hat = self.v_Emb / (1 - beta2 ** self.t)
        self.Emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

def train_model(X, y, d, k, lr, l2, epochs, rng, verbose=False):
    model = FMModel(d, k, rng)
    n = X.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits = model.forward(Xb)
            # BCE gradient: sigmoid(logits) - y
            probs = sigmoid(logits)
            grad_logits = (probs - yb).astype(np.float32)
            g_bias, g_W, g_Emb = model.backward(Xb, grad_logits)
            model.adam_step(g_bias, g_W, g_Emb, lr, l2)
    return model

def evaluate(model, X, y):
    logits = model.forward(X)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "Only binary classification supported"

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1
    best_cfg = None
    best_model = None
    for cfg in GRID:
        model = train_model(X_train, y_train, d, cfg["k"], cfg["lr"], cfg["l2"], cfg["epochs"], rng)
        acc = evaluate(model, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_model = model

    # Retrain on full data with best config
    final_model = train_model(X, y, d, best_cfg["k"], best_cfg["lr"], best_cfg["l2"], FINAL_EPOCHS, rng)

    # Save weights
    # Format: magic(0x464D4C31), d, k, C, bias, W[d*256], Emb[d*256*k] all float32 little-endian
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, best_cfg["k"], C)
    out += struct.pack("<f", final_model.bias)
    out += final_model.W.astype("<f4").tobytes()
    out += final_model.Emb.astype("<f4").tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *ptr = (const float *)(wb + 16);
    float bias = *ptr++;
    const float *W = ptr; ptr += (long)d * 256;
    const float *Emb = ptr;
    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(k * sizeof(float));
    float *sum_emb_sq = malloc(k * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = bias;
        memset(sum_emb, 0, k * sizeof(float));
        memset(sum_emb_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            score += W[(long)j * 256 + v];
            const float *e = Emb + ((long)j * 256 + v) * k;
            for (int l = 0; l < k; l++) {
                float val = e[l];
                sum_emb[l] += val;
                sum_emb_sq[l] += val * val;
            }
        }
        for (int l = 0; l < k; l++) {
            score += 0.5f * (sum_emb[l] * sum_emb[l] - sum_emb_sq[l]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```