```hypothesis
Family: label-conv1d. A tiny 1D convolution over learned byte embeddings captures local n-gram patterns in the 16-byte context better than a linear model on one-hot features. With embedding size 8, 16 filters of width 3, and a single linear output layer, it has ~3.5k parameters and should exceed label-linear's 33.8% accuracy while staying under 2k instructions/item. If dev accuracy stays below 34% or inference cost exceeds label-linear by >2x, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
SEED = 1
E = 8          # embedding dimension
F = 16         # number of conv filters
K = 3          # kernel size
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: (256, E)
    W_emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Conv: (F, K, E)
    W_conv = (rng.uniform(-1, 1, (F, K, E)) * math.sqrt(6.0 / (K*E))).astype(np.float32)
    b_conv = np.zeros(F, dtype=np.float32)
    # Linear: (F, C)
    W_lin = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b_lin = np.zeros(C, dtype=np.float32)

    params = [W_emb, W_conv, b_conv, W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    W_conv_flat = W_conv.reshape(F, -1)  # (F, K*E)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]  # (B, d)
            yb = y[idx]  # (B,)
            B = xb.shape[0]

            # Forward
            emb = W_emb[xb]  # (B, d, E)
            # Pad with zeros on both sides
            pad = np.pad(emb, ((0,0),(1,1),(0,0)), mode='constant')  # (B, d+2, E)
            # Unfold: for each position, take K consecutive vectors
            unfold = pad[:, np.arange(d)[:,None] + np.arange(K), :]  # (B, d, K, E)
            unfold_flat = np.ascontiguousarray(unfold.reshape(B, d, -1))  # (B, d, K*E)
            # Conv
            conv_out = unfold_flat @ W_conv_flat.T + b_conv  # (B, d, F)
            relu = np.maximum(conv_out, 0)
            pool = relu.mean(axis=1)  # (B, F)
            logits = pool @ W_lin.T + b_lin  # (B, C)

            # Loss and gradients
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # Linear gradients
            dW_lin = dlogits.T @ pool + L2 * W_lin
            db_lin = dlogits.sum(axis=0)

            # Pool gradients
            dpool = dlogits @ W_lin  # (B, F)

            # ReLU gradients
            drelu = dpool[:, None, :] / d  # (B, 1, F)
            drelu = np.broadcast_to(drelu, (B, d, F))
            drelu[conv_out <= 0] = 0

            # Conv gradients
            dW_conv_flat = drelu.transpose(2,0,1).reshape(F, -1) @ unfold_flat.reshape(-1, K*E) + L2 * W_conv_flat
            db_conv = drelu.sum(axis=(0,1))

            # Unfold gradients
            dunfold_flat = drelu @ W_conv_flat  # (B, d, K*E)
            dunfold = dunfold_flat.reshape(B, d, K, E)

            # Pad gradients
            pad_grad = np.zeros((B, d+2, E), dtype=np.float32)
            for k in range(K):
                pad_grad[:, k:k+d, :] += dunfold[:, :, k, :]
            emb_grad = pad_grad[:, 1:-1, :]  # (B, d, E)

            # Embedding gradients
            dW_emb = np.zeros((256, E), dtype=np.float32)
            np.add.at(dW_emb, xb.ravel(), emb_grad.reshape(-1, E))
            dW_emb += L2 * W_emb

            # Reshape conv gradient to match W_conv shape
            dW_conv = dW_conv_flat.reshape(F, K, E)

            grads = [dW_emb, dW_conv, db_conv, dW_lin, db_lin]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    MAGIC = 0x434F4E56  # "CONV"
    out = struct.pack('<5i', MAGIC, d, E, F, C)
    out += W_emb.astype('<f4').tobytes()
    out += W_conv.astype('<f4').tobytes()   # (F, K, E) in C-order
    out += b_conv.astype('<f4').tobytes()
    out += W_lin.astype('<f4').tobytes()
    out += b_lin.astype('<f4').tobytes()
    open(sys.argv[4], 'wb').write(out)

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
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2