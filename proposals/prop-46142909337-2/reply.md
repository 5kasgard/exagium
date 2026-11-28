```hypothesis
Family: label-cnn1d (1D convolutional network with character embeddings).
Why: Character-level language modeling benefits from local n-gram features. A CNN with kernel size 3 captures trigrams, which are strong predictors in text. The model is small (~18k parameters) and inference is fast (mostly matrix multiplies). It should beat the linear model (which only sees per-position unigrams) by capturing local context.
Failure: If dev accuracy is not above 0.34 (linear's 0.3375) or if inference cost exceeds linear's 7360 Ir/item.
```

```python
"""CNN family for next-character prediction: embedding -> conv1d(k=3) -> ReLU -> global max pool -> linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, Adam lr 3e-3, L2 1e-4, batch 64, epochs min(100, ceil(600000/n)).
Exported weights: int32 magic 0x434E4E31, d, E, C, K, num_classes; float32 embedding[256*E], conv_w[C*E*K], conv_b[C], lin_w[num_classes*C], lin_b[num_classes]."""
import json, math, struct, sys

import numpy as np

E = 32          # embedding dimension
C = 64          # conv output channels
K = 3           # kernel size
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy(logits, y):
    """logits: (N, C), y: (N,) int. Returns loss, dlogits (N, C)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    dlogits = probs
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    return loss, dlogits


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 16
    assert card["classes"] == 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # He initialization for embeddings and conv, Xavier for linear
    emb = (rng.standard_normal((256, E)) * math.sqrt(2.0 / 256)).astype(np.float32)
    conv_w = (rng.standard_normal((C, E * K)) * math.sqrt(2.0 / (E * K))).astype(np.float32)
    conv_b = np.zeros(C, np.float32)
    lin_w = (rng.standard_normal((65, C)) * math.sqrt(2.0 / C)).astype(np.float32)
    lin_b = np.zeros(65, np.float32)

    params = [emb, conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            N = xb.shape[0]

            # forward
            emb_x = emb[xb]                     # (N, 16, E)
            # pad for convolution (pad 1 on each side)
            emb_pad = np.pad(emb_x, ((0,0),(1,1),(0,0)), mode='constant')  # (N, 18, E)
            # im2col: (N, 16, 3, E) -> (N*16, 3*E)
            cols = np.lib.stride_tricks.as_strided(
                emb_pad, shape=(N, 16, K, E),
                strides=(emb_pad.strides[0], emb_pad.strides[1], emb_pad.strides[1], emb_pad.strides[2])
            ).reshape(N * 16, E * K)
            conv_out = cols @ conv_w.T + conv_b   # (N*16, C)
            conv_out = conv_out.reshape(N, 16, C)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.max(axis=1)       # (N, C)
            pool_argmax = relu_out.argmax(axis=1) # (N, C)
            logits = pool_out @ lin_w.T + lin_b   # (N, 65)

            loss, dlogits = softmax_cross_entropy(logits, yb)

            # backward
            dlin_w = dlogits.T @ pool_out
            dlin_b = dlogits.sum(axis=0)
            dpool_out = dlogits @ lin_w           # (N, C)

            # max pool backward
            drelu_out = np.zeros_like(relu_out)
            drelu_out[np.arange(N)[:, None], pool_argmax, np.arange(C)] = dpool_out

            # relu backward
            dconv_out = drelu_out * (conv_out > 0)
            dconv_out_flat = dconv_out.reshape(N * 16, C)

            dconv_w = dconv_out_flat.T @ cols
            dconv_b = dconv_out_flat.sum(axis=0)
            dcols = dconv_out_flat @ conv_w       # (N*16, 3*E)
            dcols = dcols.reshape(N, 16, K, E)

            # col2im
            demb_pad = np.zeros_like(emb_pad)
            for k in range(K):
                demb_pad[:, k:k+16, :] += dcols[:, :, k, :]
            demb_x = demb_pad[:, 1:17, :]         # (N, 16, E)

            # embedding backward
            demb = np.zeros_like(emb)
            np.add.at(demb, xb.flatten(), demb_x.reshape(-1, E))

            # L2 regularization gradients
            demb += L2 * emb
            dconv_w += L2 * conv_w
            dlin_w += L2 * lin_w

            grads = [demb, dconv_w, dconv_b, dlin_w, dlin_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<6i", magic, 16, E, C, K, 65)
    out = header
    out += emb.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += lin_w.astype("<f4").tobytes()
    out += lin_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* CNN inference for next-character prediction: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 24 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], K = hd[4], num_classes = hd[5];
    if (d != 16 || K != 3 || num_classes != 65 || E <= 0 || C <= 0) {
        fprintf(stderr, "bad weight dims\n"); return 3;
    }
    long expected = 24 + 4L * (256L * E + C * E * K + C + num_classes * C + num_classes);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *conv_w = emb + 256L * E;
    const float *conv_b = conv_w + C * E * K;
    const float *lin_w = conv_b + C;
    const float *lin_b = lin_w + num_classes * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb_seq = malloc((d + 2) * E * sizeof(float));  // padded sequence (18, E)
    float *conv_out = malloc(d * C * sizeof(float));       // (16, C)
    float *pool_out = malloc(C * sizeof(float));
    float *logits = malloc(num_classes * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding lookup + padding (zeros at start and end)
        memset(emb_seq, 0, (d + 2) * E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            memcpy(emb_seq + (j + 1) * E, e, E * sizeof(float));
        }

        // convolution: for each position 0..15, each output channel
        for (int p = 0; p < d; p++) {
            for (int c = 0; c < C; c++) {
                float sum = conv_b[c];
                const float *w = conv_w + c * E * K;
                for (int kk = 0; kk < K; kk++) {
                    const float *inp = emb_seq + (p + kk) * E;
                    for (int e = 0; e < E; e++) {
                        sum += inp[e] * w[kk * E + e];
                    }
                }
                conv_out[p * C + c] = sum > 0.0f ? sum : 0.0f;  // ReLU
            }
        }

        // global max pooling over positions
        for (int c = 0; c < C; c++) {
            float maxv = conv_out[c];
            for (int p = 1; p < d; p++) {
                float v = conv_out[p * C + c];
                if (v > maxv) maxv = v;
            }
            pool_out[c] = maxv;
        }

        // linear layer
        for (int cl = 0; cl < num_classes; cl++) {
            float sum = lin_b[cl];
            const float *w = lin_w + cl * C;
            for (int c = 0; c < C; c++) {
                sum += pool_out[c] * w[c];
            }
            logits[cl] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int cl = 1; cl < num_classes; cl++) {
            if (logits[cl] > logits[best]) best = (unsigned char)cl;
        }
        put_rec(stdout, &best, 1);
    }

    free(emb_seq); free(conv_out); free(pool_out); free(logits);
    if (x) free(x);
    return 0;
}
```