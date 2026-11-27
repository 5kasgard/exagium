```hypothesis
Family: label-trie, a longest-match n-gram trie with additive smoothing.
Why: Character-level language modeling benefits from variable-length contexts; a trie captures all observed suffixes up to length 16 with smoothing, providing strong non-linear context conditioning at minimal inference cost (≤16 steps).
Failure: If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds label-linear by >2x.
```

```python
import json, struct, sys

import numpy as np

# candidate alphas for additive smoothing (tuned on last 20% of train.in)
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_trie(X, y, C):
    """Build trie from examples. Returns (counts, children).
    counts: list of list[int] length C per node.
    children: list of list[int] length 256 per node (since input bytes 0..255)."""
    counts = [[0] * C]
    children = [[-1] * 256]
    for xi, yi in zip(X, y):
        node = 0
        # traverse reversed context (most recent first)
        for b in reversed(xi):
            counts[node][yi] += 1
            nxt = children[node][b]
            if nxt == -1:
                nxt = len(counts)
                children[node][b] = nxt
                counts.append([0] * C)
                children.append([-1] * 256)
            node = nxt
        counts[node][yi] += 1  # leaf node
    return counts, children

def evaluate_trie(counts, children, X, y, alpha, C):
    """Return accuracy on (X,y) using additive smoothing with alpha."""
    # precompute best class per node
    best = []
    for cts in counts:
        total = sum(cts)
        denom = total + alpha * C
        best_class = 0
        best_score = -1.0
        for c in range(C):
            score = (cts[c] + alpha) / denom
            if score > best_score:
                best_score = score
                best_class = c
        best.append(best_class)
    correct = 0
    for xi, yi in zip(X, y):
        node = 0
        pred = best[node]
        for b in reversed(xi):
            nxt = children[node][b]
            if nxt == -1:
                break
            node = nxt
            pred = best[node]
        if pred == yi:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = len(X)
    tune_size = n // 5  # last 20%
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    # choose alpha on tuning split
    best_alpha = ALPHAS[0]
    best_acc = -1.0
    for alpha in ALPHAS:
        counts, children = build_trie(train_X, train_y, C)
        acc = evaluate_trie(counts, children, tune_X, tune_y, alpha, C)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # rebuild on all data with best alpha
    counts, children = build_trie(X, y, C)
    # compute final best classes
    best_class = []
    for cts in counts:
        total = sum(cts)
        denom = total + best_alpha * C
        best_c = 0
        best_s = -1.0
        for c in range(C):
            s = (cts[c] + best_alpha) / denom
            if s > best_s:
                best_s = s
                best_c = c
        best_class.append(best_c)

    # write weights.bin
    # magic "TRIE" = 0x45495254 (little-endian)
    magic = 0x45495254
    num_nodes = len(counts)
    out = struct.pack("<I I f", magic, num_nodes, best_alpha)
    for i in range(num_nodes):
        out += struct.pack("<B", best_class[i])
        for b in range(256):
            out += struct.pack("<i", children[i][b])
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x45495254) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t num_nodes = *(int32_t*)(w + 4);
    float alpha = *(float*)(w + 8); (void)alpha;
    long expected = 12 + (long)num_nodes * (1 + 256 * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const unsigned char *best_class = w + 12;
    const int32_t *children = (const int32_t*)(w + 12 + num_nodes);  // after best_class array

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        int node = 0;
        unsigned char pred = best_class[node];
        for (int j = 15; j >= 0; j--) {
            int32_t nxt = children[node * 256 + xi[j]];
            if (nxt == -1) break;
            node = nxt;
            pred = best_class[node];
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```