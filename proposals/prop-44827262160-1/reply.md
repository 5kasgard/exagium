```hypothesis
Family: label-adaboost-stumps (multiclass AdaBoost with decision stumps).
Why: Optdigits pixels are informative individually; stumps capture single-pixel thresholds non-linearly.
Boosting combines many weak learners for strong accuracy, often beating linear models on tabular data.
Inference is extremely fast (T~200 stumps * 64 features = few hundred ops/item) and model tiny (<2KB).
Failure mode: If pixel interactions are crucial (e.g., strokes), stumps may underfit vs. trees/MLPs; dev accuracy < 0.961 would show it doesn't beat label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
CANDIDATE_T = [50, 100, 150, 200, 300]
MAGIC = 0x424F4F53  # 'BOOS'

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_adaboost(X, y, T, C):
    n, d = X.shape
    w = np.ones(n, dtype=np.float64) / n
    stumps = []
    total_weight = 1.0
    for _ in range(T):
        cnt = np.zeros((d, 17, C), dtype=np.float64)
        for j in range(d):
            indices = X[:, j] * C + y
            cnt_flat = np.bincount(indices, weights=w, minlength=17*C)
            cnt[j] = cnt_flat.reshape(17, C)
        best_err = 1.0
        best_j = best_th = best_cl = best_cg = -1
        for j in range(d):
            cum = np.cumsum(cnt[j], axis=0)
            total = cum[-1]
            for th in range(17):
                left = cum[th]
                right = total - left
                cl = int(np.argmax(left))
                cg = int(np.argmax(right))
                correct = left[cl] + right[cg]
                err = (total_weight - correct) / total_weight
                if err < best_err - 1e-12:
                    best_err = err
                    best_j, best_th, best_cl, best_cg = j, th, cl, cg
                elif abs(err - best_err) < 1e-12:
                    if (j < best_j or (j == best_j and (th < best_th or
                        (th == best_th and (cl < best_cl or (cl == best_cl and cg < best_cg)))))):
                        best_j, best_th, best_cl, best_cg = j, th, cl, cg
        if best_err >= 1.0 - 1.0/C:
            break
        alpha = math.log((1.0 - best_err) / best_err) + math.log(C - 1)
        pred = np.where(X[:, best_j] <= best_th, best_cl, best_cg)
        wrong = (pred != y)
        w[wrong] *= math.exp(alpha)
        w_sum = w.sum()
        if w_sum == 0:
            break
        w /= w_sum
        total_weight = w_sum
        stumps.append((best_j, best_th, best_cl, best_cg, alpha))
    return stumps

def evaluate(X, y, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float64)
    for (j, th, cl, cg, alpha) in stumps:
        pred = np.where(X[:, j] <= th, cl, cg)
        for i in range(n):
            scores[i, pred[i]] += alpha
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == y)

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
    best_T = CANDIDATE_T[0]
    best_acc = -1.0
    for T in CANDIDATE_T:
        stumps = train_adaboost(X_train, y_train, T, C)
        acc = evaluate(X_tune, y_tune, stumps, C)
        if acc > best_acc:
            best_acc = acc
            best_T = T
    stumps = train_adaboost(X, y, best_T, C)
    out = struct.pack("<4i", MAGIC, len(stumps), d, C)
    for (j, th, cl, cg, alpha) in stumps:
        out += struct.pack("<BBBBf", j, th, cl, cg, alpha)
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
    if (nw < 16 || hd[0] != 0x424F4F53) { fprintf(stderr, "bad weights\n"); return 3; }
    int T = hd[1], d = hd[2], C = hd[3];
    if (C < 1 || C > 256 || T < 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)T * 8;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    unsigned char *stump_data = (unsigned char *)(hd + 4);
    uint8_t *feature = malloc(T);
    uint8_t *threshold = malloc(T);
    uint8_t *class_le = malloc(T);
    uint8_t *class_gt = malloc(T);
    float *alpha = malloc(T * sizeof(float));
    for (int t = 0; t < T; t++) {
        feature[t] = stump_data[t*8 + 0];
        threshold[t] = stump_data[t*8 + 1];
        class_le[t] = stump_data[t*8 + 2];
        class_gt[t] = stump_data[t*8 + 3];
        memcpy(&alpha[t], &stump_data[t*8 + 4], sizeof(float));
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int t = 0; t < T; t++) {
            uint8_t val = xi[feature[t]];
            if (val <= threshold[t]) scores[class_le[t]] += alpha[t];
            else scores[class_gt[t]] += alpha[t];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```