```hypothesis
We propose a small convolutional neural network (label-cnn) with one 3×3 convolutional layer (16 filters, padding=1) followed by a fully connected layer. CNNs exploit the 2D spatial structure of 8×8 handwritten digits, which should yield higher accuracy than linear models on this task. If dev accuracy does not exceed label-linear (0.9613) or inference cost is significantly higher than label-mlp16, the family does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np
from numpy.lib.stride_tricks import as_strided

SEED = 1
LR = 1e-3
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def im2col(images, kernel_h, kernel_w, pad=1, stride=1):
    N, H, W, C = images.shape
    out_h = (H + 2*pad - kernel_h) // stride + 1
    out_w = (W + 2*pad - kernel_w) // stride + 1
    padded = np.pad(images, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    s0, s1, s2, s3 = padded.strides
    patches = as_strided(padded,
                         shape=(N, out_h, out_w, C, kernel_h, kernel_w),
                         strides=(s0, s1*stride, s2*stride, s3, s1, s2),
                         writeable=False)
    return patches.reshape(N, out_h*out_w, C*kernel_h*kernel_w)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X_bytes.shape[0]
    X = X_bytes.astype(np.float32) / 16.0
    X = X.reshape(n, 8, 8, 1)
    
    rng = np.random.default_rng(SEED)
    
    conv_filters = 16
    kernel = 3
    conv_W = rng.normal(0, np.sqrt(2.0/(1*kernel*kernel)), (conv_filters, 1, kernel, kernel)).astype(np.float32)
    conv_b = np.zeros(conv_filters, np.float32)
    fc_in = 8*8*conv_filters
    fc_W = rng.normal(0, np.sqrt(2.0/fc_in), (C, fc_in)).astype(np.float32)
    fc_b = np.zeros(C, np.float32)
    
    params = [conv_W, conv_b, fc_W, fc_b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X[idx]
            yb = y[idx]
            b = x.shape[0]
            
            col = im2col(x, kernel, kernel, pad=1, stride=1)
            conv_out = col @ conv_W.reshape(conv_filters, -1).T + conv_b
            relu_out = np.maximum(conv_out, 0)
            flat = relu_out.reshape(b, -1)
            logits = flat @ fc_W.T + fc_b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            
            grad_logits = probs
            grad_logits[np.arange(b), yb] -= 1.0
            grad_logits /= b
            
            grad_fc_W = grad_logits.T @ flat + L2 * fc_W
            grad_fc_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ fc_W
            
            grad_relu = grad_flat.reshape(b, 64, 16) * (conv_out > 0)
            grad_relu_flat = grad_relu.reshape(-1, conv_filters)
            col_flat = col.reshape(-1, 9)
            grad_conv_W = grad_relu_flat.T @ col_flat + L2 * conv_W.reshape(conv_filters, -1)
            grad_conv_b = grad_relu.sum(axis=(0,1))
            
            t += 1
            grads = [grad_conv_W.reshape(conv_filters,1,kernel,kernel), grad_conv_b, grad_fc_W, grad_fc_b]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1**t)
                v_hat = v[i] / (1 - beta2**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)
    
    magic = 0x434E4E01
    out = struct.pack("<5i", magic, conv_filters, kernel, fc_in, C)
    out += conv_W.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += fc_W.astype("<f4").tobytes()
    out += fc_b.astype("<f4").tobytes()
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E01) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int conv_filters = hd[1];
    int kernel = hd[2];
    int fc_in = hd[3];
    int C = hd[4];
    if (conv_filters != 16 || kernel != 3 || fc_in != 1024 || C != 10) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    size_t expected = 20 + (16*1*3*3 + 16 + 10*1024 + 10) * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *ptr = (float *)(wb + 20);
    float *conv_W = ptr; ptr += 16*9;
    float *conv_b = ptr; ptr += 16;
    float *fc_W = ptr; ptr += 10*1024;
    float *fc_b = ptr;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float conv_out[8][8][16];
    float flat[1024];
    float logits[10];
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        
        float input[8][8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                input[r][c] = xi[r*8 + c] / 16.0f;
            }
        }
        
        for (int f = 0; f < 16; f++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = conv_b[f];
                    for (int dr = 0; dr < 3; dr++) {
                        for (int dc = 0; dc < 3; dc++) {
                            int ir = r + dr - 1;
                            int ic = c + dc - 1;
                            if (ir >= 0 && ir < 8 && ic >= 0 && ic < 8) {
                                sum += input[ir][ic] * conv_W[f*9 + dr*3 + dc];
                            }
                        }
                    }
                    conv_out[r][c][f] = sum > 0 ? sum : 0;
                }
            }
        }
        
        int idx = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                for (int f = 0; f < 16; f++) {
                    flat[idx++] = conv_out[r][c][f];
                }
            }
        }
        
        for (int c = 0; c < 10; c++) {
            float sum = fc_b[c];
            for (int k = 0; k < 1024; k++) {
                sum += flat[k] * fc_W[c*1024 + k];
            }
            logits[c] = sum;
        }
        
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```