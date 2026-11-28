```hypothesis
Family: label-stumps (gradient boosted decision stumps).
Why: Optdigits pixels (0-16) have strong local interactions; stumps capture these non-linearly with ~200 simple threshold checks, faster than label-linear's 640 table lookups. Deterministic GB with fixed seed, tuned on last 20% train.
Failure: dev accuracy < 0.9613 or Ir/item > 5000.
```
```python
import json, struct, sys
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

def train_gb(X, y, C, d, num_rounds, lr):
    n = X.shape[0]
    bias = np.zeros(C, np.float32)
    for c in range(C):
        pc = (y == c).mean()
        bias[c] = np.log(pc + 1e-9)
    
    scores = np.tile(bias, (n, 1))
    stumps = []
    
    for r in range(num_rounds):
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        for c in range(C):
            g = (y == c).astype(np.float32) - probs[:, c]
            g2 = np.dot(g, g)
            
            best_loss = np.inf
            best_feat = 0
            best_thresh = 0
            best_w = 0.0
            
            for feat in range(d):
                vals = X[:, feat]
                sum_g = np.bincount(vals, weights=g, minlength=17)
                cnt = np.bincount(vals, minlength=17)
                
                cum_sum = np.cumsum(sum_g[::-1])[::-1]
                cum_cnt = np.cumsum(cnt[::-1])[::-1]
                
                num = cum_sum[1:17]
                den = cum_cnt[1:17]
                valid = den > 0
                if not valid.any():
                    continue
                w_vals = np.full(16, np.nan, dtype=np.float32)
                w_vals[valid] = num[valid] / den[valid]
                loss_vals = g2 - num*num/den
                loss_vals[~valid] = np.inf
                idx = np.argmin(loss_vals)
                if loss_vals[idx] < best_loss:
                    best_loss = loss_vals[idx]
                    best_feat = feat
                    best_thresh = idx
                    best_w = w_vals[idx]
            
            stumps.append((best_feat, best_thresh, c, best_w * lr))
            mask = (X[:, best_feat] > best_thresh).astype(np.float32)
            scores[:, c] += lr * best_w * mask
    
    return stumps, bias

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    
    best_acc = -1
    best_params = (20, 0.1)
    
    for num_rounds in [10, 15, 20, 25, 30]:
        for lr in [0.05, 0.1, 0.15, 0.2]:
            stumps, bias = train_gb(X_train, y_train, C, d, num_rounds, lr)
            scores = np.tile(bias, (len(X_tune), 1))
            for feat, thresh, c, w in stumps:
                mask = (X_tune[:, feat] > thresh).astype(np.float32)
                scores[:, c] += w * mask
            preds = scores.argmax(axis=1)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (num_rounds, lr)
    
    print(f"Best: rounds={best_params[0]}, lr={best_params[1]}, tune_acc={best_acc:.4f}", file=sys.stderr)
    
    num_rounds, lr = best_params
    stumps, bias = train_gb(X, y, C, d, num_rounds, lr)
    
    S = len(stumps)
    out = struct.pack("<5i", 0x4D555453, d, C, S, num_rounds)
    for feat, thresh, c, w in stumps:
        out += struct.pack("<3Bf", feat, thresh, c, w)
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4D555453) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], S = hd[3], rounds = hd[4];
    if (d != 64 || C != 10 || S < 0 || rounds < 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + (long)S * 7 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    
    unsigned char *stumps = wb + 20;
    float *bias = (float *)(wb + 20 + (long)S * 7);
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, bias, sizeof(float) * C);
        for (int s = 0; s < S; s++) {
            unsigned char feat = stumps[s * 7 + 0];
            unsigned char thresh = stumps[s * 7 + 1];
            unsigned char cls = stumps[s * 7 + 2];
            float w; memcpy(&w, stumps + s * 7 + 3, sizeof(float));
            if (xi[feat] > thresh) scores[cls] += w;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```