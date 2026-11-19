```hypothesis
Family: label-embed-mlp. Learns a dense embedding per (position, character) followed by an optional ReLU hidden layer.
Why: One-hot linear (label-linear) uses 16×65=1040 sparse features; dense embeddings of size D=16–32 give 256–512 features, capturing character similarities and reducing parameters. A small hidden layer (H=32–64) adds non-linearity for context interactions. Both should improve accuracy over linear at similar or lower inference cost (dense dot products vs. sparse gather-add).
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ 7359 (label-linear's cost), the hypothesis is disproven.
```

```python
"""label-embed-mlp: dense embeddings + optional 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Tunes (D, H) on last 20% of train.in, then trains on all.
Weights format: magic 0x454D424C, d, D, H, C, vocab_size; 256-byte token map; embeddings[d][vocab_size][D];
if H>0: W1[d*D][H], b1[H], W2[H][C], b2[C]; else: W1[d*D][C], b1[C]. All float32 little-endian."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
D_GRID = [16, 32]
H_GRID = [0, 32, 64]
SEED = 1
LR, L2, BATCH, EPOCHS = 3e-3, 1e-4, 64, 20
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_token_map(class_chars):
    """Map each byte value (0-255) to token index 0..vocab_size-1, 255 for unknown."""
    token_map = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        token_map[ord(ch)] = idx
    return token_map

def tokenize(X_bytes, token_map):
    """Convert (n, d) uint8 bytes to (n, d) token indices."""
    return token_map[X_bytes]

def init_params(d, vocab_size, D, H, C, rng):
    """Initialize embeddings and MLP weights."""
    scale_emb = math.sqrt(6.0 / vocab_size)
    embeddings = (rng.uniform(-1, 1, (d, vocab_size, D)) * scale_emb).astype(np.float32)
    if H:
        scale1 = math.sqrt(6.0 / (d * D))
        W1 = (rng.uniform(-1, 1, (d * D, H)) * scale1).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        scale2 = math.sqrt(6.0 / H)
        W2 = (rng.uniform(-1, 1, (H, C)) * scale2).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return embeddings, W1, b1, W2, b2
    else:
        scale1 = math.sqrt(6.0 / (d * D))
        W1 = (rng.uniform(-1, 1, (d * D, C)) * scale1).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        return embeddings, W1, b1, None, None

def forward(embeddings, W1, b1, W2, b2, tokens, H):
    """tokens: (n, d) int32. Returns logits (n, C)."""
    n, d = tokens.shape
    D = embeddings.shape[2]
    # Gather embeddings: (n, d, D)
    x = embeddings[np.arange(d), tokens.T].transpose(1, 0, 2)  # (n, d, D)
    x = x.reshape(n, d * D)  # (n, d*D)
    if H is not None and H > 0:
        h = np.maximum(x @ W1 + b1, 0, dtype=np.float32)
        logits = h @ W2 + b2
    else:
        logits = x @ W1 + b1
    return logits

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of loss w.r.t logits: (P - one_hot) / n."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def backward(embeddings, W1, b1, W2, b2, tokens, grad_logits, H, L2):
    """Compute gradients for all parameters."""
    n, d = tokens.shape
    D = embeddings.shape[2]
    # Gradient w.r.t x (n, d*D)
    if H is not None and H > 0:
        # grad_logits (n, C) -> grad_h (n, H)
        grad_h = grad_logits @ W2.T
        # ReLU derivative
        h_pre = (embeddings[np.arange(d), tokens.T].transpose(1, 0, 2).reshape(n, d * D) @ W1 + b1)
        grad_h *= (h_pre > 0).astype(np.float32)
        # Gradients for W2, b2
        h = np.maximum(h_pre, 0, dtype=np.float32)
        grad_W2 = h.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(axis=0)
        # Gradients for W1, b1
        x = embeddings[np.arange(d), tokens.T].transpose(1, 0, 2).reshape(n, d * D)
        grad_W1 = x.T @ grad_h + L2 * W1
        grad_b1 = grad_h.sum(axis=0)
        # Gradient w.r.t x
        grad_x = grad_h @ W1.T
    else:
        grad_W1 = None
        grad_b1 = None
        grad_W2 = None
        grad_b2 = None
        x = embeddings[np.arange(d), tokens.T].transpose(1, 0, 2).reshape(n, d * D)
        grad_W1 = x.T @ grad_logits + L2 * W1
        grad_b1 = grad_logits.sum(axis=0)
        grad_x = grad_logits @ W1.T

    # Gradient w.r.t embeddings: scatter-add grad_x into (d, vocab_size, D)
    grad_emb = np.zeros_like(embeddings)
    grad_x_reshaped = grad_x.reshape(n, d, D)
    for j in range(d):
        np.add.at(grad_emb[j], tokens[:, j], grad_x_reshaped[:, j, :])
    grad_emb += L2 * embeddings

    if H is not None and H > 0:
        return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2
    else:
        return grad_emb, grad_W1, grad_b1, None, None

def train_model(X_tokens, y, d, vocab_size, C, D, H, rng, n_train):
    """Train on given data, return parameters and Adam states."""
    embeddings, W1, b1, W2, b2 = init_params(d, vocab_size, D, H, C, rng)
    params = [embeddings, W1, b1]
    if H > 0:
        params += [W2, b2]
    # Adam states
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            tokens_batch = X_tokens[idx]
            y_batch = y[idx]
            logits = forward(embeddings, W1, b1, W2, b2, tokens_batch, H)
            grad_logits = softmax_cross_entropy_grad(logits, y_batch)
            grads = backward(embeddings, W1, b1, W2, b2, tokens_batch, grad_logits, H, L2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = states[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return embeddings, W1, b1, W2, b2

def evaluate(embeddings, W1, b1, W2, b2, X_tokens, y, H):
    logits = forward(embeddings, W1, b1, W2, b2, X_tokens, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def write_weights(path, d, D, H, C, vocab_size, token_map, embeddings, W1, b1, W2, b2):
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    header = struct.pack("<6i", magic, d, D, H, C, vocab_size)
    token_map_bytes = token_map.tobytes()
    emb_bytes = embeddings.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    out = header + token_map_bytes + emb_bytes + W1_bytes + b1_bytes
    if H > 0:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(out)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]
    vocab_size = len(class_chars)  # 65

    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)

    # Token map
    token_map = build_token_map(class_chars)
    X_tokens = tokenize(X_bytes, token_map)

    # Split tuning (last 20% in file order)
    n_tune = int(n * TUNING_FRAC)
    n_train = n - n_tune
    X_train, X_tune = X_tokens[:n_train], X_tokens[n_train:]
    y_train, y_tune = y[:n_train], y[n_train:]

    # Grid search
    best_acc = -1
    best_params = None
    best_D, best_H = D_GRID[0], H_GRID[0]
    rng_base = np.random.default_rng(SEED)

    for D in D_GRID:
        for H in H_GRID:
            # Deterministic seed per config
            seed = SEED + D * 1000 + H
            rng = np.random.default_rng(seed)
            embeddings, W1, b1, W2, b2 = train_model(X_train, y_train, d, vocab_size, C, D, H, rng, n_train)
            acc = evaluate(embeddings, W1, b1, W2, b2, X_tune, y_tune, H)
            print(f"D={D} H={H} tune_acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_D, best_H = D, H
                best_params = (embeddings, W1, b1, W2, b2)

    print(f"Best: D={best_D} H={best_H} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best hyperparameters
    seed = SEED + best_D * 1000 + best_H
    rng = np.random.default_rng(seed)
    embeddings, W1, b1, W2, b2 = train_model(X_tokens, y, d, vocab_size, C, best_D, best_H, rng, n)

    # Write final weights
    write_weights(sys.argv[4], d, best_D, best_H, C, vocab_size, token_map,
                  embeddings, W1, b1, W2, b2)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic 0x454D424C, d, D, H, C, vocab_size (int32 each);
 * 256-byte token_map (uint8, 255=unknown); embeddings[d][vocab_size][D] (float32);
 * if H>0: W1[d*D][H], b1[H], W2[H][C], b2[C]; else: W1[d*D][C], b1[C]. All little-endian. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4], vocab_size = hd[5];
    if (d != 16 || C != 65 || vocab_size != 65 || D <= 0 || H < 0 || H > 256) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    const unsigned char *token_map = (const unsigned char *)(hd + 6);
    const float *emb = (const float *)(token_map + 256);
    long emb_count = (long)d * vocab_size * D;
    const float *W1 = emb + emb_count;
    const float *b1 = W1 + (long)d * D * (H ? H : C);
    const float *W2 = b1 + (H ? H : C);
    const float *b2 = W2 + (long)(H ? H : 0) * C;

    // Validate total size
    long expected = 24 + 256 + 4L * (emb_count + (long)d * D * (H ? H : C) + (H ? H : C)
                    + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_vec = malloc(sizeof(float) * d * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding vector
        for (int j = 0; j < d; j++) {
            int tok = token_map[xi[j]];
            if (tok >= vocab_size) tok = 0; // fallback, should not happen
            const float *e = emb + ((long)j * vocab_size + tok) * D;
            memcpy(x_vec + j * D, e, sizeof(float) * D);
        }

        if (H) {
            // hidden = ReLU(x_vec @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int w = 0; w < d * D; w++) sum += x_vec[w] * W1[(long)w * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // scores = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                scores[c] = sum;
            }
        } else {
            // scores = x_vec @ W1 + b1
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int w = 0; w < d * D; w++) sum += x_vec[w] * W1[(long)w * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```