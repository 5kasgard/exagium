```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (size 8) followed by a one-hidden-layer ReLU MLP (128 units). Captures byte similarities and non-linear interactions missing in linear model, while being cheaper than GBT. Fails if dev accuracy <= 0.3375 (label-linear) or CPU cost >= 131k Ir/item (label-gbdt).
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 8          # embedding size
H = 128        # hidden size
LR = 0.001
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split tuning (last 20%)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    embeddings = rng.normal(0, 1/math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    # W1: (d*E, H)
    W1 = rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        batch = X_batch.shape[0]
        concat = np.zeros((batch, d*E), dtype=np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = embeddings[j, X_batch[:, j]]
        hidden = concat @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU in-place
        logits = hidden @ W2 + b2
        return concat, hidden, logits

    def loss_grad(logits, y_batch):
        batch = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        return probs / batch

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    for epoch in range(max_epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            batch = len(idx)

            concat, hidden, logits = forward(X_batch)
            grad_logits = loss_grad(logits, y_batch)

            # Gradients for W2, b2
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            # Gradients for hidden
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden == 0] = 0  # ReLU derivative

            # Gradients for W1, b1
            grad_W1 = concat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)

            # Gradients for embeddings
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                grad_concat_j = grad_hidden[:, j*E:(j+1)*E]
                np.add.at(grad_embeddings[j], X_batch[:, j], grad_concat_j)
            grad_embeddings += L2 * embeddings

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<5i", magic, d, E, H, C)
    out += embeddings.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embeddings = (const float *)(hd + 5);
    const float *W1 = embeddings + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + (j * 256 + xi[j]) * E;
            memcpy(concat + j * E, emb, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k_idx = 0; k_idx < d * E; k_idx++) {
                sum += concat[k_idx] * W1[k_idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(concat);
    free(hidden);
    free(logits);
    return 0;
}
```