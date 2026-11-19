```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of prototypes per class (here 1-5) and classifies by nearest prototype (Euclidean distance). On 8x8 optdigits, prototypes capture stroke variations better than a linear boundary, while inference stays fast: P*d multiply-adds (P~30, d=64). We tune prototypes/class, learning rate, and epochs on the last 20% of train.in. Expected to beat label-linear (96.1%) and approach label-mlp16 (97.7%) at lower instruction count. Failure mode: if accuracy stays ≤96.1%, the data is essentially linearly separable or prototypes overfit.
```
```python
import json, math, struct, sys, hashlib
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, n_proto_per_class, lr, epochs, rng):
    n, d = X.shape
    C = int(y.max()) + 1
    prototypes = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        perm = rng.permutation(len(idx))
        chosen = idx[perm[:min(n_proto_per_class, len(idx))]]
        for i in chosen:
            prototypes.append(X[i].copy())
            proto_labels.append(c)
    prototypes = np.array(prototypes, dtype=np.float32)
    proto_labels = np.array(proto_labels, dtype=np.int64)
    P = len(prototypes)
    proto_norms = np.sum(prototypes * prototypes, axis=1)
    
    for epoch in range(epochs):
        curr_lr = lr * (1.0 - epoch / epochs)
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            dots = prototypes @ x
            scores = proto_norms - 2 * dots
            best = np.argmin(scores)
            best_label = proto_labels[best]
            true_label = y[i]
            if best_label == true_label:
                prototypes[best] += curr_lr * (x - prototypes[best])
            else:
                prototypes[best] -= curr_lr * (x - prototypes[best])
            proto_norms[best] = np.sum(prototypes[best] * prototypes[best])
    return prototypes, proto_labels

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    X = X_bytes.astype(np.float32)
    
    n = len(X)
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    
    best_acc = -1.0
    best_params = None
    best_prototypes = None
    best_labels = None
    
    grid = []
    for n_proto in [1, 2, 3, 5]:
        for lr in [0.1, 0.05, 0.01]:
            for epochs in [10, 20, 50]:
                grid.append((n_proto, lr, epochs))
    
    for n_proto, lr, epochs in grid:
        trial_seed = int(hashlib.sha256(f"{SEED},{n_proto},{lr},{epochs}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(trial_seed)
        protos, labels = lvq_train(X_train, y_train, n_proto, lr, epochs, rng)
        proto_norms = np.sum(protos * protos, axis=1)
        dots = X_tune @ protos.T
        scores = proto_norms - 2 * dots
        preds = labels[np.argmin(scores, axis=1)]
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = (n_proto, lr, epochs)
            best_prototypes = protos.copy()
            best_labels = labels.copy()
    
    n_proto, lr, epochs = best_params
    final_seed = SEED + 12345
    rng = np.random.default_rng(final_seed)
    protos, labels = lvq_train(X, y, n_proto, lr, epochs, rng)
    
    P = len(protos)
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, P, d, C)
    for i in range(P):
        out += struct.pack("<B", labels[i])
        out += protos[i].astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int P = hd[1], d = hd[2], C = hd[3];
    if (P < 1 || d != 64 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    unsigned char *w = (unsigned char *)(hd + 4);
    long expected = 16L + (long)P * (1 + (long)d * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    unsigned char *proto_labels = malloc(P);
    float *prototypes = malloc((long)P * d * sizeof(float));
    for (int i = 0; i < P; i++) {
        proto_labels[i] = w[i * (1 + d * 4)];
        float *p = (float *)(w + i * (1 + d * 4) + 1);
        memcpy(prototypes + (long)i * d, p, d * sizeof(float));
    }
    
    float *proto_norms = malloc(P * sizeof(float));
    for (int i = 0; i < P; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = prototypes[(long)i * d + j];
            sum += v * v;
        }
        proto_norms[i] = sum;
    }
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *dots = malloc(P * sizeof(float));
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            const float *proto = prototypes + (long)p * d;
            for (int j = 0; j < d; j++) {
                sum += xi[j] * proto[j];
            }
            dots[p] = sum;
        }
        float best_score = 1e30f;
        int best_proto = 0;
        for (int p = 0; p < P; p++) {
            float score = proto_norms[p] - 2.0f * dots[p];
            if (score < best_score) {
                best_score = score;
                best_proto = p;
            }
        }
        unsigned char best_class = proto_labels[best_proto];
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```