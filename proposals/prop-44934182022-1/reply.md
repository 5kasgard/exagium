```hypothesis
Propose label-stumpboost: multiclass AdaBoost (SAMME) with decision stumps on raw 0–16 pixel values.
Stumps are extremely cheap at inference (one compare + add per stump) and can model non-linear boundaries.
On optdigits, 200–300 stumps should exceed label-linear's 0.9613 accuracy while staying below its 5058 Ir/item.
If dev accuracy ≤ 0.9613 or cost ≥ 5058 Ir/item, the family fails its goal.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
MAX_T = 500
T_VALUES = [50, 100, 200, 300, 400, 500]
K = 10  # classes

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_stumps(X, y, sample_weight, max_t, rng):
    n, d = X.shape
    stumps = []  # each: (feat, thresh, left_cls, right_cls, alpha)
    for _ in range(max_t):
        best_err = 1.0
        best = None
        for feat in range(d):
            xf = X[:, feat]
            for thresh in range(16):  # 0..15, split <=thresh / >thresh
                left = xf <= thresh
                right = ~left
                lw = np.bincount(y[left], weights=sample_weight[left], minlength=K)
                rw = np.bincount(y[right], weights=sample_weight[right], minlength=K)
                lc = int(np.argmax(lw))
                rc = int(np.argmax(rw))
                err = 1.0 - (lw[lc] + rw[rc])
                if err < best_err:
                    best_err = err
                    best = (feat, thresh, lc, rc)
        if best is None or best_err >= (K - 1) / K:
            break
        feat, thresh, lc, rc = best
        alpha = math.log((1 - best_err) / best_err) + math.log(K - 1)
        pred = np.where(X[:, feat] <= thresh, lc, rc)
        sample_weight *= np.exp(alpha * (pred != y))
        sample_weight /= sample_weight.sum()
        stumps.append((feat, thresh, lc, rc, alpha))
    return stumps

def predict_stumps(X, stumps):
    n = X.shape[0]
    scores = np.zeros((n, K), dtype=np.float32)
    for feat, thresh, lc, rc, alpha in stumps:
        left = X[:, feat] <= thresh
        scores[left, lc] += alpha
        scores[~left, rc] += alpha
    return np.argmax(scores, axis=1)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)  # deterministic shuffle for tuning split? Card says last 20% in file order.
    # Use file order: last 20% as tuning
    split = int(0.8 * n)
    tr_idx, tu_idx = idx[:split], idx[split:]
    # Actually, the card says "last 20% of train.in in file order". So we should not shuffle.
    # The records are read in file order. So we just take the last 20% as tuning.
    tr_idx = np.arange(split)
    tu_idx = np.arange(split, n)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_tu, y_tu = X[tu_idx], y[tu_idx]

    # Train on train_part up to MAX_T, evaluate at T_VALUES
    sw = np.ones(len(X_tr), dtype=np.float64) / len(X_tr)
    all_stumps = []
    best_t, best_acc = 0, 0.0
    tu_scores = np.zeros((len(X_tu), K), dtype=np.float32)
    for t in range(1, MAX_T + 1):
        # one iteration
        best_err = 1.0
        best = None
        for feat in range(64):
            xf = X_tr[:, feat]
            for thresh in range(16):
                left = xf <= thresh
                right = ~left
                lw = np.bincount(y_tr[left], weights=sw[left], minlength=K)
                rw = np.bincount(y_tr[right], weights=sw[right], minlength=K)
                lc = int(np.argmax(lw))
                rc = int(np.argmax(rw))
                err = 1.0 - (lw[lc] + rw[rc])
                if err < best_err:
                    best_err = err
                    best = (feat, thresh, lc, rc)
        if best is None or best_err >= (K - 1) / K:
            break
        feat, thresh, lc, rc = best
        alpha = math.log((1 - best_err) / best_err) + math.log(K - 1)
        all_stumps.append((feat, thresh, lc, rc, alpha))
        pred = np.where(X_tr[:, feat] <= thresh, lc, rc)
        sw *= np.exp(alpha * (pred != y_tr))
        sw /= sw.sum()
        # update tuning scores incrementally
        left_tu = X_tu[:, feat] <= thresh
        tu_scores[left_tu, lc] += alpha
        tu_scores[~left_tu, rc] += alpha
        if t in T_VALUES:
            tu_pred = np.argmax(tu_scores, axis=1)
            acc = np.mean(tu_pred == y_tu)
            if acc > best_acc:
                best_acc, best_t = acc, t

    # Retrain on full data with best_t
    if best_t == 0:
        best_t = MAX_T
    sw = np.ones(n, dtype=np.float64) / n
    final_stumps = []
    for t in range(best_t):
        best_err = 1.0
        best = None
        for feat in range(64):
            xf = X[:, feat]
            for thresh in range(16):
                left = xf <= thresh
                right = ~left
                lw = np.bincount(y[left], weights=sw[left], minlength=K)
                rw = np.bincount(y[right], weights=sw[right], minlength=K)
                lc = int(np.argmax(lw))
                rc = int(np.argmax(rw))
                err = 1.0 - (lw[lc] + rw[rc])
                if err < best_err:
                    best_err = err
                    best = (feat, thresh, lc, rc)
        if best is None or best_err >= (K - 1) / K:
            break
        feat, thresh, lc, rc = best
        alpha = math.log((1 - best_err) / best_err) + math.log(K - 1)
        final_stumps.append((feat, thresh, lc, rc, alpha))
        pred = np.where(X[:, feat] <= thresh, lc, rc)
        sw *= np.exp(alpha * (pred != y))
        sw /= sw.sum()

    # Write weights.bin
    magic = 0x53545542  # 'STUB'
    d = 64
    T = len(final_stumps)
    C = K
    header = struct.pack("<4i", magic, d, T, C)
    body = bytearray()
    for feat, thresh, lc, rc, alpha in final_stumps:
        body += struct.pack("<BBBBf", feat, thresh, lc, rc, alpha)
    open(sys.argv[4], "wb").write(header + body)

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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x53545542) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    int T = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;
    int C = wb[12] | wb[13] << 8 | wb[14] << 16 | (uint32_t)wb[15] << 24;
    if (d != 64 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)T * 8;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    struct Stump { uint8_t feat, thresh, lc, rc; float alpha; };
    struct Stump *stumps = malloc(T * sizeof(struct Stump));
    unsigned char *p = wb + 16;
    for (int i = 0; i < T; i++) {
        stumps[i].feat = p[0];
        stumps[i].thresh = p[1];
        stumps[i].lc = p[2];
        stumps[i].rc = p[3];
        memcpy(&stumps[i].alpha, p + 4, 4);
        p += 8;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, sizeof(scores));
        for (int t = 0; t < T; t++) {
            uint8_t v = xi[stumps[t].feat];
            if (v <= stumps[t].thresh) scores[stumps[t].lc] += stumps[t].alpha;
            else scores[stumps[t].rc] += stumps[t].alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```