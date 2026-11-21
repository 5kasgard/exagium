```hypothesis
Family: label-embed (position-specific byte embeddings + 1-hidden-layer ReLU MLP).
Why: Character prediction benefits from learned embeddings capturing byte similarities (e.g., letters cluster) and local n-gram interactions via the hidden layer, improving over one-hot linear (33.8%) while staying far faster than GBDT (131k Ir/item).
Failure: If dev accuracy ≤ 0.34 or Ir/item > 20000, the model doesn't justify its complexity over label-linear.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 42

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
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = len(X_all)
    
    n_tune = n_all // 5
    X_train, X_tune = X_all[:-n_tune], X_all[-n_tune:]
    y_train, y_tune = y_all[:-n_tune], y_all[-n_tune:]
    n_train = len(X_train)
    
    configs = [(8, 64), (8, 32), (16, 64), (4, 64)]
    
    best_acc = -1.0
    best_params = None
    best_weights = None
    
    rng = np.random.default_rng(SEED)
    
    for E, H in configs:
        emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
        W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        
        lr = 0.01
        momentum = 0.9
        v_emb = np.zeros_like(emb)
        v_W1 = np.zeros_like(W1)
        v_b1 = np.zeros_like(b1)
        v_W2 = np.zeros_like(W2)
        v_b2 = np.zeros_like(b2)
        
        batch_size = 256
        epochs = 20
        
        for epoch in range(epochs):
            order = rng.permutation(n_train)
            for s in range(0, n_train, batch_size):
                idx = order[s:s+batch_size]
                xb = X_train[idx]
                yb = y_train[idx]
                bs = len(yb)
                
                emb_x = np.zeros((bs, d*E), np.float32)
                for j in range(d):
                    emb_x[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
                
                h = emb_x @ W1 + b1
                h_relu = np.maximum(h, 0)
                logits = h_relu @ W2 + b2
                
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(bs), yb] -= 1
                probs /= bs
                
                grad_W2 = h_relu.T @ probs
                grad_b2 = probs.sum(0)
                grad_h_relu = probs @ W2.T
                grad_h = grad_h_relu * (h > 0)
                grad_W1 = emb_x.T @ grad_h
                grad_b1 = grad_h.sum(0)
                grad_emb_x = grad_h @ W1.T
                
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    g = grad_emb_x[:, j*E:(j+1)*E]
                    np.add.at(grad_emb[j], xb[:, j], g)
                
                for param, grad, vel in [(emb, grad_emb, v_emb),
                                          (W1, grad_W1, v_W1),
                                          (b1, grad_b1, v_b1),
                                          (W2, grad_W2, v_W2),
                                          (b2, grad_b2, v_b2)]:
                    vel *= momentum
                    vel += lr * grad
                    param -= vel
        
        emb_x_tune = np.zeros((n_tune, d*E), np.float32)
        for j in range(d):
            emb_x_tune[:, j*E:(j+1)*E] = emb[j, X_tune[:, j]]
        h = emb_x_tune @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        pred = logits.argmax(1)
        acc = (pred == y_tune).mean()
        
        if acc > best_acc:
            best_acc = acc
            best_params = (E, H)
            best_weights = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())
    
    E, H = best_params
    emb, W1, b1, W2, b2 = best_weights
    
    lr = 0.01
    momentum = 0.9
    v_emb = np.zeros_like(emb)
    v_W1 = np.zeros_like(W1)
    v_b1 = np.zeros_like(b1)
    v_W2 = np.zeros_like(W2)
    v_b2 = np.zeros_like(b2)
    
    batch_size = 256
    epochs = 30
    
    for epoch in range(epochs):
        order = rng.permutation(n_all)
        for s in range(0, n_all, batch_size):
            idx = order[s:s+batch_size]
            xb = X_all[idx]
            yb = y_all[idx]
            bs = len(yb)
            
            emb_x = np.zeros((bs, d*E), np.float32)
            for j in range(d):
                emb_x[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
            
            h = emb_x @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2
            
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1
            probs /= bs
            
            grad_W2 = h_relu.T @ probs
            grad_b2 = probs.sum(0)
            grad_h_relu = probs @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = emb_x.T @ grad_h
            grad_b1 = grad_h.sum(0)
            grad_emb_x = grad_h @ W1.T
            
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                g = grad_emb_x[:, j*E:(j+1)*E]
                np.add.at(grad_emb[j], xb[:, j], g)
            
            for param, grad, vel in [(emb, grad_emb, v_emb),
                                      (W1, grad_W1, v_W1),
                                      (b1, grad_b1, v_b1),
                                      (W2, grad_W2, v_W2),
                                      (b2, grad_b2, v_b2)]:
                vel *= momentum
                vel += lr * grad
                param -= vel
    
    magic = 0x454D4245
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad params\n"); return 3; }
    
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        
        for (int j = 0; j < d; j++) {
            const float *emb_j = emb + ((long)j * 256 + xi[j]) * E;
            const float *W1_j = W1 + (long)j * E * H;
            for (int e = 0; e < E; e++) {
                float ev = emb_j[e];
                const float *w = W1_j + (long)e * H;
                for (int h = 0; h < H; h++) {
                    hidden[h] += ev * w[h];
                }
            }
        }
        
        for (int h = 0; h < H; h++) if (hidden[h] < 0) hidden[h] = 0;
        
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w[c];
        }
        
        unsigned char best = 0;
        float best_val = logits[0];
        for (int c = 1; c < C; c++) {
            if (logits[c] > best_val) {
                best_val = logits[c];
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```