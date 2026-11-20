```hypothesis
Family: label-tiedmlp. Uses shared 256×E character embeddings + 16×E positional embeddings concatenated, one ReLU layer (H), then linear to 65 classes. Ties embeddings across positions to capture character similarities (e.g., vowels) and reduce parameters vs. per-position one-hot (∼45k vs 266k params). Exports in label-linear compatible format for fast inference. Fails if dev accuracy ≤ 0.3375 (label-linear) or if parameter tying hurts context-specific patterns.
```
```python
"""label-tiedmlp: MLP with tied character embeddings across positions for label tasks with fixed-size categorical byte inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Learns shared 256xE character embeddings and dxE positional embeddings, concatenates, one ReLU layer (H), then linear to C classes.
Exports in label-linear compatible format: magic, d, H, C, T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 32        # character embedding dimension
H = 64        # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    for t in card["input"]["types"]:
        assert t == "categorical", "only categorical supported"

    n = X_bytes.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    char_emb = (rng.normal(0, 1/math.sqrt(E), (256, E))).astype(np.float32)
    pos_emb = (rng.normal(0, 1/math.sqrt(E), (d, E))).astype(np.float32)
    W1_full = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [char_emb, pos_emb, W1_full, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X_bytes[idx]
            yb = y[idx]
            B = len(yb)

            # Forward: concatenated embeddings (B, d*E)
            x_emb = np.zeros((B, d*E), np.float32)
            for j in range(d):
                x_emb[:, j*E:(j+1)*E] = char_emb[xb[:,j]] + pos_emb[j]

            h_pre = x_emb @ W1_full + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Loss gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)

            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h_pre > 0)

            grad_W1_full = x_emb.T @ grad_h_pre + L2 * W1_full
            grad_b1 = grad_h_pre.sum(0)

            grad_x_emb = grad_h_pre @ W1_full.T

            grad_char_emb = np.zeros_like(char_emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                g = grad_x_emb[:, j*E:(j+1)*E]
                np.add.at(grad_char_emb, xb[:,j], g)
                grad_pos_emb[j] = g.sum(0)

            grads = [grad_char_emb, grad_pos_emb, grad_W1_full, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export tables T[d][256][H]
    T = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        for c in range(256):
            vec = char_emb[c] + pos_emb[j]
            T[j, c] = vec @ W1_full[j*E:(j+1)*E, :]

    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-tiedmlp inference (compatible with label-linear weights format):
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```