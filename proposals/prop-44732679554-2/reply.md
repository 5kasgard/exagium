```hypothesis
Family: label-embed (shared byte embeddings + position-specific linear output). Factorizes the 16×256×65 linear weight matrix into a shared 256×D embedding and 16 position-specific D×65 classifiers (D=8), reducing parameters from ~266k to ~11k. This exploits the shared byte vocabulary across positions (TinyShakespeare uses ~65 chars) for better generalization on 24k samples, while the compact 45KB weights fit in L1 cache for faster inference than label-linear's 1MB. If dev accuracy ≤ 0.3375, the factorization is too restrictive for this task.
```
```python
"""label-embed: shared byte embeddings + position-specific linear classifiers.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16-byte input -> shared embedding (256, D) -> 16 position-specific (D, 65) + bias -> sum -> softmax.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), D=8.
Weights: magic 0x454D4245, d=16, D, C=65; float32 E[256][D], W[16][D][65], b[16][65]."""
import json, math, struct, sys

import numpy as np

D = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    
    E = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d, D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros((d, C), np.float32)
    
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)
            
            scores = np.zeros((bs, C), np.float32)
            embs = np.zeros((d, bs, D), np.float32)
            for j in range(d):
                embs[j] = E[xb[:, j]]
                scores += embs[j] @ W[j] + b[j]
            
            max_logits = scores.max(axis=1, keepdims=True)
            exp_logits = np.exp(scores - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_scores = (probs / bs).astype(np.float32)
            
            grad_E = np.zeros_like(E)
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)
            
            for j in range(d):
                grad_b[j] = grad_scores.sum(axis=0)
                grad_W[j] = embs[j].T @ grad_scores
                grad_emb = grad_scores @ W[j].T
                np.add.at(grad_E, xb[:, j], grad_emb)
            
            grad_E += L2 * E
            grad_W += L2 * W
            grad_b += L2 * b
            
            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    out = struct.pack("<4i", 0x454D4245, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed: shared byte embeddings + position-specific linear classifiers.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + (long)256 * D * 4 + (long)d * D * C * 4 + (long)d * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *E = (const float *)(hd + 4);
    const float *W = E + 256 * D;
    const float *b = W + (long)d * D * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        memset(scores, 0, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = E + (long)v * D;
            const float *w = W + (long)j * D * C;
            const float *bj = b + (long)j * C;
            
            for (int c = 0; c < C; c++) {
                float sum = bj[c];
                for (int d_ = 0; d_ < D; d_++) {
                    sum += e[d_] * w[(long)d_ * C + c];
                }
                scores[c] += sum;
            }
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```