```hypothesis
We propose label-lowrank: a low-rank factorized linear model. The label-linear baseline uses a full 256×65 weight matrix per position (16×256×65 parameters). By factorizing each position's matrix into 256×D and D×65 with D≪256 (chosen on the tuning split), we drastically reduce parameters, acting as a strong regularizer that should improve generalization on this small dataset. Inference expands the factors to the full matrix, so CPU cost is identical to label-linear. If dev accuracy does not exceed label-linear, the low-rank assumption does not hold for this task.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 30
EPOCHS_FULL = 50
CANDIDATE_D = [4, 8, 16, 32, 64]

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X_bytes[:train_size], y[:train_size]
    X_val, y_val = X_bytes[train_size:], y[train_size:]
    
    rng = np.random.default_rng(SEED)
    
    def init_params(D):
        params = []
        for j in range(d):
            E = rng.normal(0, 0.01, (256, D)).astype(np.float32)
            W = rng.normal(0, 0.01, (D, C)).astype(np.float32)
            params.append((E, W))
        b = np.zeros(C, np.float32)
        return params, b
    
    def compute_scores(X, params, b):
        B = X.shape[0]
        scores = np.zeros((B, C), np.float32)
        for j in range(d):
            E_j, W_j = params[j]
            emb = E_j[X[:, j]]
            scores += emb @ W_j
        scores += b
        return scores
    
    def loss_and_grad(X, y, params, b):
        B = X.shape[0]
        scores = compute_scores(X, params, b)
        max_s = scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        grad_scores = probs
        grad_scores[np.arange(B), y] -= 1.0
        grad_scores /= B
        grad_params = []
        for j in range(d):
            E_j, W_j = params[j]
            emb = E_j[X[:, j]]
            grad_W = emb.T @ grad_scores
            grad_E = np.zeros_like(E_j)
            for i in range(B):
                v = X[i, j]
                grad_E[v] += grad_scores[i] @ W_j.T
            grad_params.append((grad_E, grad_W))
        grad_b = grad_scores.sum(axis=0)
        for j in range(d):
            grad_params[j][0] += L2 * params[j][0]
            grad_params[j][1] += L2 * params[j][1]
        grad_b += L2 * b
        loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
        return loss, grad_params, grad_b
    
    def train_model(X_tr, y_tr, X_vl, y_vl, D, epochs):
        params, b = init_params(D)
        param_list = []
        for j in range(d):
            param_list.append(params[j][0])
            param_list.append(params[j][1])
        param_list.append(b)
        m = [np.zeros_like(p) for p in param_list]
        v = [np.zeros_like(p) for p in param_list]
        t = 0
        n_tr = X_tr.shape[0]
        for epoch in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s+BATCH]
                X_batch, y_batch = X_tr[idx], y_tr[idx]
                loss, grad_params, grad_b = loss_and_grad(X_batch, y_batch, params, b)
                t += 1
                grad_list = []
                for j in range(d):
                    grad_list.append(grad_params[j][0])
                    grad_list.append(grad_params[j][1])
                grad_list.append(grad_b)
                for i, (p, gr) in enumerate(zip(param_list, grad_list)):
                    m[i] = 0.9 * m[i] + 0.1 * gr
                    v[i] = 0.999 * v[i] + 0.001 * gr * gr
                    m_hat = m[i] / (1 - 0.9**t)
                    v_hat = v[i] / (1 - 0.999**t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                idx_p = 0
                for j in range(d):
                    params[j] = (param_list[idx_p], param_list[idx_p+1])
                    idx_p += 2
                b = param_list[-1]
        scores_val = compute_scores(X_vl, params, b)
        preds = scores_val.argmax(axis=1)
        acc = (preds == y_vl).mean()
        return params, b, acc
    
    best_D = None
    best_acc = -1
    best_params = None
    best_b = None
    for D in CANDIDATE_D:
        print(f"Trying D={D}", file=sys.stderr)
        params, b, acc = train_model(X_train, y_train, X_val, y_val, D, EPOCHS_TUNE)
        print(f"  val acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_params = params
            best_b = b
    
    print(f"Best D={best_D} with val acc={best_acc:.4f}", file=sys.stderr)
    
    params_full, b_full, _ = train_model(X_bytes, y, X_bytes, y, best_D, EPOCHS_FULL)
    
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        E_j, W_j = params_full[j]
        T[j] = E_j @ W_j
    
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b_full.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```