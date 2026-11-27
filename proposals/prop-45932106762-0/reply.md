```hypothesis
Family: label-fm (Factorization Machine). Adult has strong pairwise feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A rank-10 FM on the 14 byte fields (256 values each) adds only ~36K parameters, trains in seconds, and inference costs ~14×10=140 multiply-adds per item. It should beat label-linear's 0.8524 accuracy. If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear by >2×, the hypothesis fails.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib + numpy.
Format: int32 magic 0x464D3130, d, k, C; float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys
import numpy as np

K = 10
SEED = 42
LR = 0.02
L2_W = 1e-5
L2_V = 1e-5
BATCH = 256
MAX_EPOCHS = 30
SAMPLES = 400000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]
    num_features = d * 256
    
    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros(num_features, dtype=np.float32)
    V = (rng.normal(0, 0.01, (num_features, K)) * np.sqrt(2.0 / K)).astype(np.float32)
    
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)
            
            scores = np.full(B, w0, dtype=np.float32)
            for j in range(d):
                feat_idx = j * 256 + xb[:, j]
                scores += w[feat_idx]
            
            sum_v = np.zeros((B, K), dtype=np.float32)
            sum_v_sq = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                feat_idx = j * 256 + xb[:, j]
                Vj = V[feat_idx]
                sum_v += Vj
                sum_v_sq += Vj * Vj
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
            scores += interaction
            
            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)
            
            gw0 = grad.mean()
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            
            for j in range(d):
                feat_idx = j * 256 + xb[:, j]
                np.add.at(gw, feat_idx, grad[:, None])
            gw /= B
            gw += L2_W * w
            
            for j in range(d):
                feat_idx = j * 256 + xb[:, j]
                Vj = V[feat_idx]
                gVj = grad[:, None] * (sum_v - Vj)
                np.add.at(gV, feat_idx, gVj)
            gV /= B
            gV += L2_V * V
            
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)
    
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0 or 1).
 * Weights format: int32 magic 0x464D3130, d, k, C; float32 w0, w[d*256], V[d*256*k]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    int num_features = d * 256;
    long expected = 16 + 4L * (1L + num_features + (long)num_features * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = p + 1 + num_features;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v_sq = malloc(sizeof(float) * k);
    unsigned char out_byte;
    
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        
        float score = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }
        
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            score += w[idx];
            const float *Vj = V + (long)idx * k;
            for (int f = 0; f < k; f++) {
                float v = Vj[f];
                sum_v[f] += v;
                sum_v_sq[f] += v * v;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) {
            float sv = sum_v[f];
            inter += sv * sv - sum_v_sq[f];
        }
        score += 0.5f * inter;
        
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```