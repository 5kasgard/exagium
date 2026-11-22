```hypothesis
Family: fm-linear (Factorization Machine augmented linear model). Adds low-rank pairwise interactions via shared per-position/byte embeddings (dim k=8) to the label-linear baseline. Character prediction benefits from n-gram statistics; FM captures bigram interactions efficiently (O(d*k + C*k) vs O(d^2*C) for full cross-features). If dev accuracy doesn't exceed label-linear (0.3375), the task may not benefit from pairwise interactions or k=8 is insufficient.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50
K_CHOICES = [4, 8, 16]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, k, rng):
    W_lin = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    V_emb = (rng.uniform(-1, 1, (d, 256, k)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    U_proj = (rng.uniform(-1, 1, (C, k)) * math.sqrt(6.0 / k)).astype(np.float32)
    return [W_lin, b_lin, V_emb, U_proj]

def forward(params, X_batch):
    W_lin, b_lin, V_emb, U_proj = params
    B, d = X_batch.shape
    C = b_lin.shape[0]
    k = V_emb.shape[2]
    logits = np.broadcast_to(b_lin, (B, C)).copy()
    emb_sum = np.zeros((B, k), np.float32)
    for j in range(d):
        bytes_j = X_batch[:, j]
        logits += W_lin[j, bytes_j, :]
        emb_sum += V_emb[j, bytes_j, :]
    logits += emb_sum @ U_proj.T
    return logits, emb_sum

def backward(params, X_batch, y_batch, logits, emb_sum, lr, l2, t, state):
    W_lin, b_lin, V_emb, U_proj = params
    B, d = X_batch.shape
    C = b_lin.shape[0]
    k = V_emb.shape[2]

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(B), y_batch] -= 1.0
    probs /= B

    db_lin = probs.sum(axis=0)
    dW_lin = np.zeros_like(W_lin)
    dV_emb = np.zeros_like(V_emb)
    dU_proj = probs.T @ emb_sum

    for j in range(d):
        bytes_j = X_batch[:, j]
        np.add.at(dW_lin[j], bytes_j, probs)
        grad_V = probs @ U_proj
        np.add.at(dV_emb[j], bytes_j, grad_V)

    dW_lin += l2 * W_lin
    dV_emb += l2 * V_emb
    dU_proj += l2 * U_proj

    grads = [dW_lin, db_lin, dV_emb, dU_proj]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t += 1
    for i, (p, gr) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= beta1; m += (1 - beta1) * gr
        v *= beta2; v += (1 - beta2) * gr * gr
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return t

def evaluate(params, X, y):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        Xb = X[s:s+1024]
        yb = y[s:s+1024]
        logits, _ = forward(params, Xb)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_model(X_train, y_train, X_val, y_val, k, rng_seed):
    d = X_train.shape[1]
    C = int(y_train.max()) + 1
    rng = np.random.default_rng(rng_seed)
    params = init_params(d, C, k, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n_train = X_train.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    t = 0
    best_val = 0.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            Xb = X_train[idx]
            yb = y_train[idx]
            logits, emb_sum = forward(params, Xb)
            t = backward(params, Xb, yb, logits, emb_sum, LR, L2, t, state)
        val_acc = evaluate(params, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    if best_params is not None:
        for i in range(len(params)):
            params[i][:] = best_params[i]
    return params, best_val

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = X.shape[0]
    tune_start = int(0.8 * n)
    X_train, y_train = X[:tune_start], y[:tune_start]
    X_val, y_val = X[tune_start:], y[tune_start:]

    best_k = K_CHOICES[0]
    best_val_acc = -1
    best_params = None
    for k in K_CHOICES:
        params, val_acc = train_model(X_train, y_train, X_val, y_val, k, SEED)
        print(f"k={k}, val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_params = params

    print(f"Selected k={best_k} with val_acc={best_val_acc:.4f}", file=sys.stderr)

    rng = np.random.default_rng(SEED)
    params = init_params(d, C, best_k, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n_train = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            logits, emb_sum = forward(params, Xb)
            t = backward(params, Xb, yb, logits, emb_sum, LR, L2, t, state)

    W_lin, b_lin, V_emb, U_proj = params
    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, best_k, C)
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += V_emb.astype("<f4").tobytes()
    out += U_proj.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* FM-linear inference: ./infer weights.bin in.rec N */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected_floats = (long)d * 256 * C + C + (long)d * 256 * k + (long)C * k;
    if (nw != 16 + 4L * expected_floats) {
        fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, 16 + 4L * expected_floats);
        return 3;
    }

    const float *W_lin = (const float *)(hd + 4);
    const float *b_lin = W_lin + (long)d * 256 * C;
    const float *V_emb = b_lin + C;
    const float *U_proj = V_emb + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *emb_sum = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        memcpy(scores, b_lin, sizeof(float) * C);
        memset(emb_sum, 0, sizeof(float) * k);

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *W_row = W_lin + ((long)j * 256 + byte) * C;
            for (int c = 0; c < C; c++) scores[c] += W_row[c];

            const float *V_row = V_emb + ((long)j * 256 + byte) * k;
            for (int kk = 0; kk < k; kk++) emb_sum[kk] += V_row[kk];
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *U_row = U_proj + (long)c * k;
            for (int kk = 0; kk < k; kk++) sum += U_row[kk] * emb_sum[kk];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores);
    free(emb_sum);
    if (x) free(x);
    return 0;
}
```