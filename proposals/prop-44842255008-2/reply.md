```hypothesis
Family: label-cnn1d (1D convolution over embeddings of the 16-byte context).
Why: Character-level language modeling benefits from local n-gram features; a small CNN captures positional patterns (e.g., "the", "ing") with far fewer parameters than a full linear layer on one-hot vectors, and inference is fast (embedding lookup + fixed-size convolution + max-pool + linear).
Failure signal: If dev accuracy does not exceed label-linear's 0.3375, the CNN's inductive bias is mismatched or the model is under-trained.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
EMB_DIM = 32
CONV_FILTERS = 64
KERNEL_SIZE = 3
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
    # Build byte->class index mapping from card's class_chars
    class_chars = card["class_chars"]
    byte_to_class = np.full(256, 0, dtype=np.uint8)  # default to class 0 (newline)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Convert input bytes to class indices (0..64)
    X = byte_to_class[X_bytes]  # shape (n, d)

    # Split for validation (last 20% in file order) to pick hyperparameters, but we hardcode above.
    # Train on all data.
    rng = np.random.default_rng(SEED)

    # Parameters
    # Embedding: per position, per class -> (d, C, EMB_DIM)
    emb = (rng.uniform(-1, 1, (d, C, EMB_DIM)) * math.sqrt(6.0 / C)).astype(np.float32)
    # Conv1d: (CONV_FILTERS, EMB_DIM, KERNEL_SIZE)
    conv_w = (rng.uniform(-1, 1, (CONV_FILTERS, EMB_DIM, KERNEL_SIZE)) * math.sqrt(6.0 / (EMB_DIM * KERNEL_SIZE))).astype(np.float32)
    conv_b = np.zeros(CONV_FILTERS, dtype=np.float32)
    # Linear: (C, CONV_FILTERS)
    lin_w = (rng.uniform(-1, 1, (C, CONV_FILTERS)) * math.sqrt(6.0 / CONV_FILTERS)).astype(np.float32)
    lin_b = np.zeros(C, dtype=np.float32)

    params = [emb, conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    pad = KERNEL_SIZE // 2  # 1 for kernel=3

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]          # (B, d)
            yy = y[idx]         # (B,)
            B = x.shape[0]

            # --- Forward ---
            # Embedding lookup: (B, d, EMB_DIM)
            e = emb[np.arange(d)[None, :], x]  # (B, d, EMB_DIM)

            # Pad for convolution (zero padding on both sides)
            e_pad = np.pad(e, ((0,0), (pad,pad), (0,0)), mode='constant')  # (B, d+2, EMB_DIM)

            # Convolution: im2col style
            # We'll slide kernel over positions 0..d-1 (output length d)
            # For each position, we take e_pad[:, pos:pos+KERNEL_SIZE, :] -> (B, KERNEL_SIZE, EMB_DIM)
            # Reshape to (B, KERNEL_SIZE*EMB_DIM) and multiply by conv_w reshaped to (CONV_FILTERS, KERNEL_SIZE*EMB_DIM)
            conv_w_flat = conv_w.reshape(CONV_FILTERS, -1)  # (F, K*E)
            conv_out = np.zeros((B, d, CONV_FILTERS), dtype=np.float32)
            for pos in range(d):
                window = e_pad[:, pos:pos+KERNEL_SIZE, :].reshape(B, -1)  # (B, K*E)
                conv_out[:, pos, :] = window @ conv_w_flat.T + conv_b  # (B, F)

            # ReLU
            h = np.maximum(conv_out, 0)

            # Global max pooling over positions
            pooled = h.max(axis=1)  # (B, F)

            # Linear
            logits = pooled @ lin_w.T + lin_b  # (B, C)

            # Softmax cross-entropy loss and gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = probs / B  # (B, C)

            # --- Backward ---
            # Linear
            grad_lin_w = grad_logits.T @ pooled + L2 * lin_w
            grad_lin_b = grad_logits.sum(axis=0)
            grad_pooled = grad_logits @ lin_w  # (B, F)

            # Max pooling backward: gradient flows only to max positions
            grad_h = np.zeros_like(h)
            # For each batch and filter, find max position(s) and assign gradient
            # We'll use argmax (first max wins)
            max_pos = h.argmax(axis=1)  # (B, F)
            for b in range(B):
                for f in range(CONV_FILTERS):
                    grad_h[b, max_pos[b, f], f] = grad_pooled[b, f]

            # ReLU backward
            grad_conv_out = grad_h * (conv_out > 0)

            # Convolution backward
            grad_conv_w = np.zeros_like(conv_w)
            grad_conv_b = grad_conv_out.sum(axis=(0,1))  # (F,)
            grad_e_pad = np.zeros_like(e_pad)

            conv_w_flat = conv_w.reshape(CONV_FILTERS, -1)
            for pos in range(d):
                window = e_pad[:, pos:pos+KERNEL_SIZE, :].reshape(B, -1)  # (B, K*E)
                grad_window = grad_conv_out[:, pos, :]  # (B, F)
                grad_conv_w += grad_window.T @ window  # (F, K*E)
                grad_e_pad[:, pos:pos+KERNEL_SIZE, :] += (grad_window @ conv_w_flat).reshape(B, KERNEL_SIZE, EMB_DIM)

            grad_conv_w = grad_conv_w.reshape(CONV_FILTERS, EMB_DIM, KERNEL_SIZE) + L2 * conv_w

            # Remove padding gradient
            grad_e = grad_e_pad[:, pad:pad+d, :]  # (B, d, EMB_DIM)

            # Embedding backward: scatter add
            grad_emb = np.zeros_like(emb)
            # For each position, we have grad_e[:, pos, :] (B, EMB_DIM) and indices x[:, pos] (B,)
            for pos in range(d):
                np.add.at(grad_emb[pos], x[:, pos], grad_e[:, pos, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_conv_w, grad_conv_b, grad_lin_w, grad_lin_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- Export weights.bin ---
    # Layout:
    # magic (int32): 0x434E4E31 ('CNN1')
    # d, emb_dim, conv_filters, kernel_size, classes (5 int32)
    # mapping: 256 uint8
    # emb: float32[d, C, EMB_DIM]
    # conv_w: float32[CONV_FILTERS, EMB_DIM, KERNEL_SIZE]
    # conv_b: float32[CONV_FILTERS]
    # lin_w: float32[C, CONV_FILTERS]
    # lin_b: float32[C]
    magic = 0x434E4E31
    header = struct.pack("<6i", magic, d, EMB_DIM, CONV_FILTERS, KERNEL_SIZE, C)
    mapping_bytes = byte_to_class.tobytes()
    out = header + mapping_bytes
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 24 + 256) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], conv_filters = hd[3], kernel_size = hd[4], C = hd[5];
    if (d != 16 || emb_dim != 32 || conv_filters != 64 || kernel_size != 3 || C != 65) {
        fprintf(stderr, "unexpected architecture constants\n"); return 3;
    }

    // mapping: 256 bytes after header (6 int32 = 24 bytes)
    unsigned char *mapping = wb + 24;
    float *ptr = (float *)(wb + 24 + 256);

    // Pointers into weight arrays
    float *emb = ptr;                                      // [d][C][emb_dim]
    ptr += d * C * emb_dim;
    float *conv_w = ptr;                                   // [conv_filters][emb_dim][kernel_size]
    ptr += conv_filters * emb_dim * kernel_size;
    float *conv_b = ptr;                                   // [conv_filters]
    ptr += conv_filters;
    float *lin_w = ptr;                                    // [C][conv_filters]
    ptr += C * conv_filters;
    float *lin_b = ptr;                                    // [C]
    // ptr += C; // end

    // Validate total size
    long expected = 24 + 256
        + (long)d * C * emb_dim * 4
        + (long)conv_filters * emb_dim * kernel_size * 4
        + (long)conv_filters * 4
        + (long)C * conv_filters * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Temporary buffers
    float *e_pad = malloc((d + 2) * emb_dim * sizeof(float)); // padded embeddings for one example
    float *conv_out = malloc(d * conv_filters * sizeof(float));
    float *h = malloc(d * conv_filters * sizeof(float));
    float *pooled = malloc(conv_filters * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    int pad = kernel_size / 2;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup with padding
        memset(e_pad, 0, (d + 2) * emb_dim * sizeof(float));
        for (int pos_in = 0; pos_in < d; pos_in++) {
            unsigned char byte = xi[pos_in];
            int cls = mapping[byte];
            float *emb_row = emb + (pos_in * C + cls) * emb_dim;
            float *dst = e_pad + (pos_in + pad) * emb_dim;
            memcpy(dst, emb_row, emb_dim * sizeof(float));
        }

        // Convolution + ReLU
        for (int p = 0; p < d; p++) {
            for (int f = 0; f < conv_filters; f++) {
                float sum = conv_b[f];
                for (int ki = 0; ki < kernel_size; ki++) {
                    float *win = e_pad + (p + ki) * emb_dim;
                    float *w = conv_w + (f * emb_dim * kernel_size) + ki * emb_dim;
                    for (int e = 0; e < emb_dim; e++) {
                        sum += win[e] * w[e];
                    }
                }
                float val = sum > 0 ? sum : 0;
                conv_out[p * conv_filters + f] = val;
                h[p * conv_filters + f] = val;
            }
        }

        // Global max pooling
        for (int f = 0; f < conv_filters; f++) {
            float maxv = -1e30f;
            for (int p = 0; p < d; p++) {
                float v = h[p * conv_filters + f];
                if (v > maxv) maxv = v;
            }
            pooled[f] = maxv;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            float *w = lin_w + c * conv_filters;
            for (int f = 0; f < conv_filters; f++) {
                sum += pooled[f] * w[f];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }

    free(e_pad); free(conv_out); free(h); free(pooled); free(logits);
    return 0;
}
```