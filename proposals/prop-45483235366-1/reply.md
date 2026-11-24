```hypothesis
Family: label-gam (Generalized Additive Model). Learns a per-pixel per-value class score table (64×17×10 for optdigits) with Adam, no interactions. Same inference cost as label-linear (64×10 adds), smaller weights, captures non-linear pixel-value effects. Expects ~97% dev accuracy (beating linear's 96.1%). If dev accuracy ≤96.5%, per-pixel non-linearity is insufficient for this task.
```
```python
import json, math, struct, sys
import numpy as np

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
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Parameters: T[d][256][C], b1[C]
    T = np.zeros((d, 256, C), dtype=np.float32)
    b1 = np.zeros(C, dtype=np.float32)

    # Adam state
    m_T = np.zeros_like(T)
    v_T = np.zeros_like(T)
    m_b = np.zeros_like(b1)
    v_b = np.zeros_like(b1)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward: logits = b1 + sum_j T[j, xb[:,j]]
            logits = np.tile(b1, (B, 1))
            for j in range(d):
                logits += T[j, xb[:, j]]

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Gradients
            grad_b1 = grad_logits.sum(axis=0) + L2 * b1
            grad_T = np.zeros_like(T)
            for j in range(d):
                np.add.at(grad_T[j], xb[:, j], grad_logits)
            grad_T += L2 * T

            # Adam update
            t += 1
            m_T = 0.9 * m_T + 0.1 * grad_T
            v_T = 0.999 * v_T + 0.001 * grad_T * grad_T
            m_hat_T = m_T / (1 - 0.9**t)
            v_hat_T = v_T / (1 - 0.999**t)
            T -= LR * m_hat_T / (np.sqrt(v_hat_T) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * grad_b1
            v_b = 0.999 * v_b + 0.001 * grad_b1 * grad_b1
            m_hat_b = m_b / (1 - 0.9**t)
            v_hat_b = v_b / (1 - 0.999**t)
            b1 -= LR * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)

    # Write weights in label-linear format (H=0)
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0) { fprintf(stderr, "weights have hidden layer, expected H=0\n"); return 3; }
    int W = C;
    long expected = 16 + 4L * ((long)d * 256 * W + W);
    if (C < 1 || C > 256 || nw != expected) {
        fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```