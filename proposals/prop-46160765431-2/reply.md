```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (size D) concatenated into a 16*D vector, followed by an optional ReLU hidden layer (size H) and a linear output layer. This captures positional interactions better than a linear model while staying small and fast. Hyperparameters (D, H) are chosen on the last 20% of train.in. Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 42
LR = 1e-3
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, D, H, epochs, seed, verbose=False):
    n, _ = X.shape
    rng = np.random.default_rng(seed)
    # Embeddings: (d, 256, D)
    E = rng.normal(0, 1.0 / math.sqrt(D), size=(d, 256, D)).astype(np.float32)
    if H:
        W1 = rng.normal(0, math.sqrt(2.0 / (d * D)), size=(d * D, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W1, b1, W2, b2]
    else:
        W2 = rng.normal(0, math.sqrt(2.0 / (d * D)), size=(d * D, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_loss = float('inf')
    wait = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Embedding lookup
            z = np.zeros((len(yb), d * D), np.float32)
            for j in range(d):
                z[:, j*D:(j+1)*D] = E[j, xb[:, j]]
            # Forward
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W2 + b2
            # Softmax + cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()
            epoch_loss += loss * len(yb)
            # Backward
            grad_logits = probs.copy()
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)
            if H:
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_h = grad_logits @ W2.T
                grad_a = grad_h * (a > 0)
                grad_W1 = z.T @ grad_a + L2 * W1
                grad_b1 = grad_a.sum(0)
                grad_z = grad_a @ W1.T
            else:
                grad_W2 = z.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_z = grad_logits @ W2.T
            # Embedding gradients
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_z[:, j*D:(j+1)*D])
            grad_E += L2 * E
            # Adam update
            grads = [grad_E] + ([grad_W1, grad_b1, grad_W2, grad_b2] if H else [grad_W2, grad_b2])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        epoch_loss /= n
        if verbose:
            print(f"  epoch {epoch+1}: loss={epoch_loss:.4f}")
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return params, best_loss

def evaluate(X, y, d, C, D, H, params):
    E = params[0]
    if H:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W2, b2 = params[1], params[2]
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb, yb = X[s:s+1024], y[s:s+1024]
        z = np.zeros((len(yb), d * D), np.float32)
        for j in range(d):
            z[:, j*D:(j+1)*D] = E[j, xb[:, j]]
        if H:
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
        else:
            logits = z @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    # Grid search on tuning split
    best_acc = -1
    best_cfg = None
    best_params = None
    for D in [8, 16, 32]:
        for H in [0, 32, 64, 128]:
            print(f"Trying D={D}, H={H}...", file=sys.stderr)
            params, _ = train_model(train_X, train_y, d, C, D, H, 20, SEED)
            acc = evaluate(tune_X, tune_y, d, C, D, H, params)
            print(f"  tune acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, H)
                best_params = params
    D, H = best_cfg
    print(f"Best: D={D}, H={H}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    final_params, _ = train_model(X, y, d, C, D, H, MAX_EPOCHS, SEED)

    # Write weights.bin
    # Format: magic, d, D, H, C (int32), then embeddings (d*256*D float32),
    # then if H: W1 (d*D*H), b1 (H), W2 (H*C), b2 (C)
    # else: W2 (d*D*C), b2 (C)
    magic = 0x454D4244  # 'EMBD' little-endian
    E = final_params[0]
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        expected = 16 + 4 * (d * 256 * D + d * D * H + H + H * C + C)
        out = struct.pack("<5i", magic, d, D, H, C)
        out += E.astype("<f4").tobytes()
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W2, b2 = final_params[1], final_params[2]
        expected = 16 + 4 * (d * 256 * D + d * D * C + C)
        out = struct.pack("<5i", magic, d, D, H, C)
        out += E.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    if len(out) != expected:
        print(f"Weight size mismatch: {len(out)} vs {expected}", file=sys.stderr)
        sys.exit(1)
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L * (d * 256 * D + (H ? (d * D * H + H + H * C + C) : (d * D * C + C)));
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + d * 256 * D;
    const float *b1 = W1 + (H ? d * D * H : 0);
    const float *W2 = b1 + (H ? H : 0);
    const float *b2 = W2 + (H ? H * C : d * D * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * D);
    float *h = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = E + (j * 256 + xi[j]) * D;
            memcpy(z + j * D, e, D * sizeof(float));
        }
        if (H) {
            // Hidden layer: ReLU(z @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < d * D; v++) sum += z[v] * W1[v * H + u];
                h[u] = sum > 0 ? sum : 0;
            }
            // Output layer
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
                logits[c] = sum;
            }
        } else {
            // Linear output
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int v = 0; v < d * D; v++) sum += z[v] * W2[v * C + c];
                logits[c] = sum;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```