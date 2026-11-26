```hypothesis
Family: label-embed (positional embeddings + 1-hidden-layer MLP).
Why: Character-level sequences benefit from dense embeddings that capture similarity between characters (e.g., letters vs punctuation) and positional context, which one-hot linear models cannot. A small MLP adds non-linear interactions. This should surpass label-linear (0.3375) and approach label-gbdt (0.3923) with lower inference cost than tree ensembles.
Failure: If dev accuracy ≤ 0.34 or inference cost exceeds label-linear by >2×, the family is not justified.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
VAL_FRACTION = 0.2

# Hyperparameter grid for tuning split
GRID = [(4, 32), (8, 32), (8, 64), (16, 64)]  # (E, H)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def evaluate(X, y, emb, W1, b1, W2, b2, H):
    n = X.shape[0]
    d = X.shape[1]
    E = emb.shape[2]
    C = b2.shape[0] if H else b1.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = slice(s, min(s+BATCH, n))
        xb = X[idx]
        yb = y[idx]
        # embeddings
        z = np.zeros((len(yb), d*E), np.float32)
        for j in range(d):
            z[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
        if H:
            h = np.maximum(z @ W1 + b1, 0)
            logits = h @ W2 + b2
        else:
            logits = z @ W1 + b1
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_on_split(X, y, X_val, y_val, d, C, E, H, epochs):
    rng = np.random.default_rng(SEED)
    n, k = X.shape
    # init embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256*E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32) if H else np.zeros((d*E, C), np.float32)
    b1 = np.zeros(H if H else C, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
    else:
        W2 = b2 = None

    params = [emb, W1, b1]
    if H: params += [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            z = np.zeros((len(yb), d*E), np.float32)
            for j in range(d):
                z[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W1 + b1
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # backward
            if H:
                grad_W2 = h.T @ g + L2 * W2
                grad_b2 = g.sum(0)
                gh = g @ W2.T * (a > 0)
                grad_W1 = z.T @ gh + L2 * W1
                grad_b1 = gh.sum(0)
                # embedding gradients
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    grad_emb[j, xb[:, j]] += gh[:, j*E:(j+1)*E]
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W1 = z.T @ g + L2 * W1
                grad_b1 = g.sum(0)
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    grad_emb[j, xb[:, j]] += g[:, j*E:(j+1)*E]
                grads = [grad_emb, grad_W1, grad_b1]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W1, b1, W2, b2

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_all)
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]

    best_acc = -1
    best_cfg = None
    best_weights = None
    for E, H in GRID:
        print(f"Trying E={E}, H={H}", file=sys.stderr)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_size))
        emb, W1, b1, W2, b2 = train_on_split(X_train, y_train, X_val, y_val, d, C, E, H, epochs)
        acc = evaluate(X_val, y_val, emb, W1, b1, W2, b2, H)
        print(f"  val acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (E, H)
            best_weights = (emb, W1, b1, W2, b2)

    print(f"Best: E={best_cfg[0]}, H={best_cfg[1]}, val acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    E, H = best_cfg
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W1, b1, W2, b2 = train_on_split(X_all, y_all, X_val, y_val, d, C, E, H, epochs)

    # Export weights
    # Header: magic, d, E, H, C (5 int32)
    MAGIC = 0x454D4244  # "EMBD"
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # MLP
    mlp_bytes = W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        mlp_bytes += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    out = header + emb_bytes + mlp_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed: positional embeddings + 1-hidden-layer MLP (or linear if H=0)
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * (H ? H : C) + (H ? H : C) + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * (H ? H : C);
    const float *W2 = H ? b1 + (H ? H : C) : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * E);
    float *h = H ? malloc(sizeof(float) * H) : NULL;
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(z + j * E, e, E * sizeof(float));
        }
        if (H) {
            // W1: (d*E, H), b1: H
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int w = 0; w < d * E; w++) sum += z[w] * W1[(long)w * H + u];
                h[u] = sum > 0.0f ? sum : 0.0f;
            }
            // W2: (H, C), b2: C
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
                s[c] = sum;
            }
        } else {
            // linear: W1: (d*E, C), b1: C
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int w = 0; w < d * E; w++) sum += z[w] * W1[(long)w * C + c];
                s[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```