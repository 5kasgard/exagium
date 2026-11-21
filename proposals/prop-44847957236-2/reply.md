```hypothesis
We propose "label-embmlp": per-position byte embeddings (dim 16) concatenated and fed to a 128-unit ReLU hidden layer, then 65-class output. This captures positional interactions with fewer parameters than linear one-hot (85k vs 266k), improving generalization on the small tinyshakespeare dataset. If dev accuracy ≤ 0.34 or inference cost > 2× label-linear, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
E = 16          # embedding dimension
H = 128         # hidden size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all data
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = X.shape[0]
    # Use all data for training (hyperparameters fixed, no tuning)
    train_X, train_y = X, y
    n = train_X.shape[0]
    
    rng = np.random.default_rng(SEED)
    
    # Initialize parameters
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, np.sqrt(2.0 / (d*E)), size=(d*E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = train_X[idx]  # (B, d)
            yb = train_y[idx]  # (B,)
            B = xb.shape[0]
            
            # Forward: gather embeddings for all positions
            # xb.T is (d, B), advanced indexing gives (d, B, E)
            emb_gathered = emb[np.arange(d)[:, None], xb.T]  # (d, B, E)
            concat = emb_gathered.transpose(1, 0, 2).reshape(B, d*E)  # (B, d*E)
            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2
            
            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B  # (B, C)
            
            # Backward
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T * (hidden_pre > 0)
            dW1 = concat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dconcat = dhidden @ W1.T  # (B, d*E)
            demb = np.zeros_like(emb)
            dconcat_3d = dconcat.reshape(B, d, E)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dconcat_3d[:, j, :])
            demb += L2 * emb
            
            grads = [demb, dW1, db1, dW2, db2]
            
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    # Write weights
    magic = 0x454D4231  # "EMB1"
    out = struct.pack("<5i", magic, d, E, H, C)
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        // Build concatenated embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        
        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) {
                sum += concat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            scores[c] = sum;
        }
        
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    
    free(concat);
    free(hidden);
    free(scores);
    return 0;
}
```