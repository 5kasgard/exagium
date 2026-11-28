```hypothesis
Family: label-hashngram (hashed n-gram linear model).
Why: Character-level language modeling benefits from short-range dependencies (n-grams). A linear model on hashed 1/2/3-grams captures these interactions with fixed memory and fast inference, bridging the gap between label-linear (1-grams only) and label-gbdt (non-linear trees).
Failure signal: If dev accuracy <= 0.3375 (label-linear) or inference cost > 20k Ir/item.
```
```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: hashed 1-grams (16 positions), 2-grams (15), 3-grams (14) -> 45 n-grams per example.
Hash table size D=8192. Trained with SGD+momentum, weight decay, deterministic seed."""
import json, struct, sys
import numpy as np

D = 8192          # hash table size (power of 2)
SEED = 12345
LR = 0.05
MOMENTUM = 0.9
L2 = 1e-5
BATCH = 512
EPOCHS = 10

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    C = card["classes"]
    d = card["input"]["bytes"]  # 16
    
    # Build byte->class mapping
    byte_to_class = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx
    
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    X = byte_to_class[X_bytes]  # (N, 16) values 0..64
    
    # N-gram specification: (length, num_positions)
    ngram_spec = [(1, 16), (2, 15), (3, 14)]
    total_ngrams = sum(p for _, p in ngram_spec)  # 45
    
    # Generate seeds for hashing
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(0, 2**32, size=total_ngrams, dtype=np.uint32)
    
    def compute_features(X_batch):
        B = X_batch.shape[0]
        feat = np.zeros((B, total_ngrams), dtype=np.int32)
        col = 0
        for length, num_pos in ngram_spec:
            for pos in range(num_pos):
                h = seeds[col].astype(np.uint64)
                for k in range(length):
                    h = (h * 31 + X_batch[:, pos + k].astype(np.uint64)) & 0xFFFFFFFF
                feat[:, col] = (h & (D - 1)).astype(np.int32)
                col += 1
        return feat
    
    W = np.zeros((D, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    vW = np.zeros_like(W)
    vb = np.zeros_like(b)
    
    N = X.shape[0]
    for epoch in range(EPOCHS):
        order = np.random.default_rng(SEED + epoch).permutation(N)
        for start in range(0, N, BATCH):
            idx = order[start:start+BATCH]
            Xb = X[idx]
            yb = y[idx]
            feat = compute_features(Xb)
            
            Wb = W[feat]
            scores = Wb.sum(axis=1) + b
            
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            
            grad_W = np.zeros((D, C), dtype=np.float32)
            for f in range(total_ngrams):
                indices = feat[:, f]
                np.add.at(grad_W, indices, probs)
            
            grad_b = probs.sum(axis=0)
            
            W *= (1.0 - LR * L2)
            vW = MOMENTUM * vW + LR * grad_W
            vb = MOMENTUM * vb + LR * grad_b
            W -= vW
            b -= vb
    
    out = bytearray()
    out += struct.pack("<4s", b"HNGM")
    out += struct.pack("<ii", D, C)
    out += struct.pack("<i", len(ngram_spec))
    for length, num_pos in ngram_spec:
        out += struct.pack("<ii", length, num_pos)
    out += seeds.astype("<u4").tobytes()
    out += byte_to_class.astype("<u1").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Hashed n-gram linear model inference: ./infer weights.bin in.rec N */
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
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
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    
    if (nw < 4 || memcmp(wb, "HNGM", 4) != 0) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t *p32 = (uint32_t *)(wb + 4);
    int D = p32[0];
    int C = p32[1];
    int num_types = p32[2];
    p32 += 3;
    
    struct { int len; int num_pos; } spec[10];
    for (int i = 0; i < num_types; i++) {
        spec[i].len = p32[0];
        spec[i].num_pos = p32[1];
        p32 += 2;
    }
    
    uint32_t *seeds = p32;
    int total_ngrams = 0;
    for (int i = 0; i < num_types; i++) total_ngrams += spec[i].num_pos;
    p32 += total_ngrams;
    
    uint8_t *byte_to_class = (uint8_t *)p32;
    p32 += 64;
    
    float *W = (float *)p32;
    float *b = W + (long)D * C;
    
    long expected = 4 + 4*3 + 4*num_types*2 + 4*total_ngrams + 256 + 4L*D*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    
    float *scores = malloc(C * sizeof(float));
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        
        uint8_t ctx[16];
        for (int j = 0; j < 16; j++) {
            uint8_t cls = byte_to_class[xi[j]];
            if (cls == 255) { fprintf(stderr, "unknown byte %d at pos %d\n", xi[j], j); return 3; }
            ctx[j] = cls;
        }
        
        for (int c = 0; c < C; c++) scores[c] = b[c];
        
        int seed_idx = 0;
        for (int t = 0; t < num_types; t++) {
            int len = spec[t].len;
            int num_pos = spec[t].num_pos;
            for (int pos_ng = 0; pos_ng < num_pos; pos_ng++) {
                uint32_t h = seeds[seed_idx++];
                for (int k = 0; k < len; k++) {
                    h = h * 31 + ctx[pos_ng + k];
                }
                int idx = h & (D - 1);
                const float *wrow = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += wrow[c];
            }
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        
        put_rec(stdout, &best, 1);
    }
    
    return 0;
}
```