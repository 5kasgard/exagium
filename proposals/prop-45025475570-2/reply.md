```hypothesis
Family: label-embed-mlp. Each of the 16 context bytes is mapped to a learned embedding (dimension E), concatenated, passed through a ReLU hidden layer (size H), then a linear output to 65 classes. This captures position interactions that a linear model cannot, while inference stays fast (embedding lookups + two matvecs). It should beat label-linear (0.3375) on dev accuracy. If dev accuracy ≤ 0.3375 or inference cost (Ir/item) exceeds label-linear by >20%, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning split (E, H)
GRID = [(4, 16), (8, 32), (16, 64)]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # same as label-linear

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, E, H, epochs, rng):
    n, d = X.shape
    # Embeddings: d positions, 256 categories, E dims
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # Hidden layer: (d*E) -> H
    W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H -> C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # Gather embeddings: (batch, d, E)
        h = emb[np.arange(d)[None, :], Xb.T].transpose(1, 0, 2)  # (batch, d, E)
        h = h.reshape(batch, d*E)  # (batch, d*E)
        # Hidden layer
        h = h @ W1 + b1
        h = np.maximum(h, 0)  # ReLU
        # Output layer
        logits = h @ W2 + b2
        return logits, h

    def softmax_cross_entropy_grad(logits, yb):
        batch = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), yb] -= 1.0
        return probs / batch  # (batch, C)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h = forward(Xb)
            g = softmax_cross_entropy_grad(logits, yb)  # (batch, C)
            # Backprop output layer
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(axis=0)
            # Backprop hidden layer
            gh = g @ W2.T
            gh *= (h > 0)  # ReLU grad
            gW1 = Xb_emb.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            # Backprop embeddings
            # Xb_emb is (batch, d, E) before flatten
            Xb_emb = emb[np.arange(d)[None, :], Xb.T].transpose(1, 0, 2)
            gemb = np.zeros_like(emb)
            for j in range(d):
                # gh_j = gh[:, j*E:(j+1)*E]  but easier: we have gh (batch, H), need to multiply by W1 slices
                # Actually gradient wrt embedding at position j: Xb_emb[:,j,:] -> gh via W1[j*E:(j+1)*E]
                W1_j = W1[j*E:(j+1)*E]  # (E, H)
                gemb_j = Xb_emb[:, j, :].T @ gh @ W1_j.T  # (E, batch) @ (batch, H) @ (H, E) -> (E, E)? Wait.
                # Let's do it properly: embedding gradient for each example: for position j, byte b, the gradient is sum over hidden units of gh * W1_j.
                # We'll accumulate per category.
                # Simpler: loop over batch (small) or use vectorized scatter.
                pass
            # Given time, we'll use a simpler approach: flatten embeddings into one big matrix and treat as linear layer with one-hot.
            # But we already have a working linear model; we need embeddings to be learned.
            # Let's change strategy: use the same one-hot encoding as label-linear but with a hidden layer.
            # Actually the label-linear already does one-hot and linear. We want embeddings to reduce dimension.
            # We'll implement a simpler but correct backprop for embeddings using loops over batch and positions (batch=64, d=16, E<=16, H<=64 -> fine).
            # However, due to time constraints in this answer, we'll fall back to a proven architecture: one-hot + one hidden layer (like label-mlp16 but with tuned H).
            # But the hypothesis is about embeddings. We must deliver a working train.py.
            # Given the complexity, I'll provide a train.py that uses one-hot encoding and a single hidden layer (like label-mlp16) but with hyperparameter search on H.
            # That is a different family: label-mlp-tuned. But the hypothesis said embeddings.
            # To satisfy the requirement, I'll write a train.py that does embeddings with correct backprop using vectorized operations.
            # Let's do it properly with numpy advanced indexing.

    # Due to the length limit, I'll provide a complete but simpler train.py that uses one-hot + hidden layer (like label-mlp16) but searches H on tuning split.
    # This is a valid new family: label-mlp-tuned.
    pass

# The above is incomplete. Let's write a complete, correct train.py for a one-hidden-layer MLP on one-hot inputs (like label-mlp16) but with H chosen on tuning split.
# This is a new family because label-mlp16 fixes H=16; we tune H.

# Actually, the existing families include label-mlp16 and label-mlp2l. We need a *different* family.
# We'll do embeddings but with a correct implementation.

# Given the time, I'll write a train.py that implements embeddings with correct backprop using vectorized scatter-add.
# We'll use the fact that d=16, E small. We can compute embedding gradients by:
# For each position j, we have indices Xb[:,j] (batch). The gradient for emb[j] is a (256, E) matrix where each row b gets sum of gh_i * W1_j for examples where Xb_i,j == b.
# We can do this with np.add.at.

# Let's write the complete train.py now.

import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
# Grid for (E, H)
GRID = [(4, 16), (8, 32), (16, 64)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_epoch(X, y, emb, W1, b1, W2, b2, rng, epoch, t, state):
    n, d = X.shape
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s+BATCH]
        Xb, yb = X[idx], y[idx]
        batch = Xb.shape[0]
        # Forward
        # Embeddings: (batch, d, E)
        h_emb = emb[np.arange(d)[None, :], Xb.T].transpose(1, 0, 2)
        h_flat = h_emb.reshape(batch, d*E)  # but E not in scope; we need d and E from emb shape
        # We'll pass d, E, H, C as arguments.
    # This is getting too long. Let's write a clean version.

# Given the constraints, I'll provide a complete train.py for the embedding MLP with correct backprop.
# We'll assume the following architecture:
# Input: 16 bytes (0-255)
# Embedding: for each position p in 0..15, a matrix E_p of shape (256, D)
# Concatenate embeddings -> vector of size 16*D
# Hidden layer: ReLU(concat * W1 + b1) -> size H
# Output: softmax(hidden * W2 + b2) -> 65
# Loss: cross-entropy
# Optimizer: Adam

# We'll implement training with vectorized embedding gradient using np.add.at.

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1
    best_weights = None
    best_EH = None

    for E, H in GRID:
        rng = np.random.default_rng(SEED)
        # Initialize
        emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
        W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        # Train for fewer epochs on tuning
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
        for epoch in range(epochs):
            order = rng.permutation(split)
            for s in range(0, split, BATCH):
                idx = order[s:s+BATCH]
                Xb, yb = X_train[idx], y_train[idx]
                batch = Xb.shape[0]

                # Forward
                # Embeddings: (batch, d, E)
                # Use advanced indexing: emb[positions, Xb.T] -> (d, batch, E) -> transpose
                h_emb = emb[np.arange(d)[:, None], Xb.T]  # (d, batch, E)
                h_emb = h_emb.transpose(1, 0, 2)  # (batch, d, E)
                h_flat = h_emb.reshape(batch, d*E)

                hidden = h_flat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2

                # Softmax cross-entropy gradient
                logits = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(batch), yb] -= 1.0
                g = probs / batch  # (batch, C)

                # Backprop output
                gW2 = hidden_relu.T @ g + L2 * W2
                gb2 = g.sum(axis=0)

                # Backprop hidden
                gh = g @ W2.T
                gh *= (hidden > 0).astype(np.float32)  # ReLU grad

                # Backprop hidden linear
                gW1 = h_flat.T @ gh + L2 * W1
                gb1 = gh.sum(axis=0)

                # Backprop embeddings
                # gh shape (batch, H). Need gradient for each embedding matrix.
                # For position p, embedding gradient is sum over examples of gh_i * W1_p^T, where W1_p is slice of W1 for that position.
                # W1 is (d*E, H). For position p, slice W1[p*E:(p+1)*E] -> (E, H)
                gemb = np.zeros_like(emb)
                for p in range(d):
                    W1_p = W1[p*E:(p+1)*E]  # (E, H)
                    # For each example, the embedding vector is h_emb[i, p] (E). Its gradient is gh[i] @ W1_p.T (E)
                    # But we need to accumulate per category.
                    # We can compute grad per example: (batch, E) = gh @ W1_p.T
                    grad_ex = gh @ W1_p.T  # (batch, E)
                    # Now scatter-add to gemb[p] using indices Xb[:, p]
                    np.add.at(gemb[p], Xb[:, p], grad_ex)
                gemb += L2 * emb

                # Adam update
                grads = [gemb, gW1, gb1, gW2, gb2]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on validation
        # Forward pass on val
        h_emb = emb[np.arange(d)[:, None], X_val.T].transpose(1, 0, 2)
        h_flat = h_emb.reshape(-1, d*E)
        hidden = np.maximum(h_flat @ W1 + b1, 0)
        logits = hidden @ W2 + b2
        preds = logits.argmax(axis=1)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_EH = (E, H)
            best_weights = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # Retrain on full data with best hyperparameters
    E, H = best_EH
    emb, W1, b1, W2, b2 = best_weights
    rng = np.random.default_rng(SEED)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            batch = Xb.shape[0]
            h_emb = emb[np.arange(d)[:, None], Xb.T].transpose(1, 0, 2)
            h_flat = h_emb.reshape(batch, d*E)
            hidden = h_flat @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch), yb] -= 1.0
            g = probs / batch
            gW2 = hidden_relu.T @ g + L2 * W2
            gb2 = g.sum(axis=0)
            gh = g @ W2.T
            gh *= (hidden > 0).astype(np.float32)
            gW1 = h_flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            gemb = np.zeros_like(emb)
            for p in range(d):
                W1_p = W1[p*E:(p+1)*E]
                grad_ex = gh @ W1_p.T
                np.add.at(gemb[p], Xb[:, p], grad_ex)
            gemb += L2 * emb
            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    # Format: magic, d, E, H, C (int32), then embeddings (d*256*E floats), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C) all float32 little-endian.
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += emb.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    float *concat = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embeddings
        for (int p = 0; p < d; p++) {
            const float *e = emb + (p * 256 + xi[p]) * E;
            memcpy(concat + p * E, e, E * sizeof(float));
        }

        // Hidden layer: concat * W1 + b1, ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h;  // W1 is (d*E) x H, stored row-major? In train.py we did W1 shape (d*E, H) and used h_flat @ W1.
            // In train.py: W1 is (d*E, H), so element (j, h) at W1[j*H + h]? Actually numpy row-major: (d*E, H) -> W1[j, h] at index j*H + h.
            // So W1 is stored with leading dimension H.
            for (int j = 0; j < d * E; j++) {
                sum += concat[j] * W1[j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        // W2 shape (H, C) in train.py, stored row-major: W2[h*C + c]
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores); free(concat);
    if (x) free(x);
    free(hd);
    return 0;
}
```