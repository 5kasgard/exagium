```hypothesis
Family: label-rnn (vanilla RNN with tanh). A small recurrent network can capture sequential dependencies in the 16-byte context better than a linear model. It uses an embedding layer, a recurrent layer with hidden size H, and a linear output. If dev accuracy does not exceed label-linear (0.3375) or inference cost is significantly higher, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
E_CHOICES = [8, 16]
H_CHOICES = [16, 32, 64]
LR_CHOICES = [1e-3, 3e-3]
BATCH = 64
EPOCHS_TUNE = 5
EPOCHS_FULL = 20
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def evaluate(X, y, d, C, E, H, emb, W_xh, W_hh, b_h, W_out, b_out):
    n = X.shape[0]
    correct = 0
    for i in range(0, n, BATCH):
        x = X[i:i+BATCH]
        yb = y[i:i+BATCH]
        B = x.shape[0]
        h = np.zeros((B, H), np.float32)
        for t in range(d):
            e = emb[x[:, t]]
            h = np.tanh(e @ W_xh + h @ W_hh + b_h)
        logits = h @ W_out + b_out
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_model(X, y, d, C, E, H, lr, epochs, rng):
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W_xh = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    W_hh = (rng.uniform(-1, 1, (H, H)) * math.sqrt(6.0 / H)).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_out = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, W_xh, W_hh, b_h, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X[idx]
            yb = y[idx]
            B = x.shape[0]
            h_states = np.zeros((B, d, H), np.float32)
            h = np.zeros((B, H), np.float32)
            for step in range(d):
                e = emb[x[:, step]]
                a = e @ W_xh + h @ W_hh + b_h
                h = np.tanh(a)
                h_states[:, step] = h
            logits = h @ W_out + b_out
            grad_logits = softmax_cross_entropy_grad(logits, yb)

            g_emb = np.zeros_like(emb)
            g_W_xh = np.zeros_like(W_xh)
            g_W_hh = np.zeros_like(W_hh)
            g_b_h = np.zeros_like(b_h)
            g_W_out = np.zeros_like(W_out)
            g_b_out = np.zeros_like(b_out)

            g_W_out = h_states[:, -1].T @ grad_logits
            g_b_out = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W_out.T

            for step in reversed(range(d)):
                h_t = h_states[:, step]
                h_prev = h_states[:, step-1] if step > 0 else np.zeros((B, H), np.float32)
                e_t = emb[x[:, step]]
                grad_a = grad_h * (1 - h_t * h_t)
                g_W_xh += e_t.T @ grad_a
                g_W_hh += h_prev.T @ grad_a
                g_b_h += grad_a.sum(axis=0)
                grad_e = grad_a @ W_xh.T
                grad_h = grad_a @ W_hh.T
                for b in range(B):
                    g_emb[x[b, step]] += grad_e[b]

            grads = [g_emb, g_W_xh, g_W_hh, g_b_h, g_W_out, g_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]
    split = int(0.8 * n_all)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_config = None

    for E in E_CHOICES:
        for H in H_CHOICES:
            for lr in LR_CHOICES:
                params = train_model(X_train, y_train, d, C, E, H, lr, EPOCHS_TUNE, rng)
                emb, W_xh, W_hh, b_h, W_out, b_out = params
                acc = evaluate(X_tune, y_tune, d, C, E, H, emb, W_xh, W_hh, b_h, W_out, b_out)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (E, H, lr)
    print(f"Best config: E={best_config[0]}, H={best_config[1]}, lr={best_config[2]}, tune_acc={best_acc:.4f}")

    E, H, lr = best_config
    params = train_model(X_all, y_all, d, C, E, H, lr, EPOCHS_FULL, rng)
    emb, W_xh, W_hh, b_h, W_out, b_out = params

    magic = 0x524E4E31
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W_xh.astype("<f4").tobytes()
    out += W_hh.astype("<f4").tobytes()
    out += b_h.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 20 + 4L * (256L * E + (long)E * H + (long)H * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W_xh = emb + 256L * E;
    const float *W_hh = W_xh + (long)E * H;
    const float *b_h = W_hh + (long)H * H;
    const float *W_out = b_h + H;
    const float *b_out = W_out + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *e = emb + (long)xi[step] * E;
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k2 = 0; k2 < E; k2++) sum += e[k2] * W_xh[(long)k2 * H + j];
                for (int k2 = 0; k2 < H; k2++) sum += h[k2] * W_hh[(long)k2 * H + j];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_out[(long)j * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```