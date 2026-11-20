```hypothesis
Family: label-embmlp (shared byte embeddings + positional embeddings + one hidden ReLU layer).
Why: The linear model (label-linear) uses per-position one-hot embeddings of size 65, ignoring interactions between positions. A shared byte embedding captures character similarity across positions, while positional embeddings retain order. A hidden layer adds non-linearity to model context interactions. This should improve accuracy over linear with moderate compute.
Failure: If dev accuracy <= 0.3375 (label-linear) or CPU cost per item exceeds 2x label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid (E, H)
GRID = [(32, 64), (32, 128), (64, 64), (64, 128)]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 5
EPOCHS_FULL = 20

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
    d = card["input"]["bytes"]
    C = card["classes"]
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n_total = X.shape[0]
    n_tune = n_total // 5
    n_train = n_total - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    rng = np.random.default_rng(SEED)

    def train_eval(E, H, X_tr, y_tr, X_te, y_te, epochs):
        scale = math.sqrt(6.0 / (256 + E))
        byte_emb = rng.uniform(-scale, scale, (256, E)).astype(np.float32)
        pos_emb = rng.uniform(-scale, scale, (d, E)).astype(np.float32)
        scale1 = math.sqrt(6.0 / (d*E + H))
        W1 = rng.uniform(-scale1, scale1, (d*E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        scale2 = math.sqrt(6.0 / (H + C))
        W2 = rng.uniform(-scale2, scale2, (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)

        params = [byte_emb, pos_emb, W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        n_tr = X_tr.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s+BATCH]
                xb = X_tr[idx]
                yb = y_tr[idx]
                B = xb.shape[0]

                concat = np.zeros((B, d*E), np.float32)
                for j in range(d):
                    concat[:, j*E:(j+1)*E] = byte_emb[xb[:, j]] + pos_emb[j]

                hidden = concat @ W1 + b1
                np.maximum(hidden, 0, out=hidden)

                logits = hidden @ W2 + b2

                max_logits = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                grad_logits = probs
                grad_logits[np.arange(B), yb] -= 1.0
                grad_logits /= B

                grad_W2 = hidden.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden *= (hidden > 0)
                grad_W1 = concat.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)
                grad_concat = grad_hidden @ W1.T

                grad_byte_emb = np.zeros_like(byte_emb)
                grad_pos_emb = np.zeros_like(pos_emb)
                for j in range(d):
                    g = grad_concat[:, j*E:(j+1)*E]
                    np.add.at(grad_byte_emb, xb[:, j], g)
                    grad_pos_emb[j] = g.sum(axis=0)

                grad_W1 += L2 * W1
                grad_b1 += L2 * b1
                grad_W2 += L2 * W2
                grad_b2 += L2 * b2
                grad_byte_emb += L2 * byte_emb
                grad_pos_emb += L2 * pos_emb

                grads = [grad_byte_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9**t)
                    v_hat = v / (1 - 0.999**t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        B = X_te.shape[0]
        concat = np.zeros((B, d*E), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = byte_emb[X_te[:, j]] + pos_emb[j]
        hidden = np.maximum(concat @ W1 + b1, 0)
        logits = hidden @ W2 + b2
        preds = logits.argmax(axis=1)
        acc = (preds == y_te).mean()
        return acc, params

    best_acc = -1.0
    best_params = None
    best_EH = None
    for E, H in GRID:
        acc, params = train_eval(E, H, X_train, y_train, X_tune, y_tune, EPOCHS_TUNE)
        print(f"E={E}, H={H}, tune_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_EH = (E, H)

    print(f"Best: E={best_EH[0]}, H={best_EH[1]}, acc={best_acc:.4f}", file=sys.stderr)
    E, H = best_EH
    _, final_params = train_eval(E, H, X, y, X[:1], y[:1], EPOCHS_FULL)
    byte_emb, pos_emb, W1, b1, W2, b2 = final_params

    magic = 0x454D424D
    out = struct.pack("<5i", magic, d, E, H, C)
    out += byte_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20L + (256L * E + d * E + (long)d * E * H + H + (long)H * C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *byte_emb = (float *)(wb + 20);
    float *pos_emb = byte_emb + 256 * E;
    float *W1 = pos_emb + d * E;
    float *b1 = W1 + (long)d * E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + xi[j] * E;
            const float *pe = pos_emb + j * E;
            float *out = concat + j * E;
            for (int e = 0; e < E; e++) out[e] = be[e] + pe[e];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < d * E; e++) sum += concat[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```