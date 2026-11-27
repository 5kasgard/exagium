```hypothesis
Propose label-cnn: a tiny CNN with one 3x3 convolutional layer (8 filters), ReLU, 2x2 max pooling, and a fully connected layer to 10 classes. It exploits the 2D spatial structure of 8x8 digits, which linear models ignore, and should beat label-linear (0.9613) while keeping inference cost near label-mlp16 (~9k Ir/item). Failure if dev accuracy ≤ label-linear or instructions/item > 15000.
```

```python
import json, math, struct, sys
import numpy as np

H_CHOICES = [8, 16]
LR_CHOICES = [1e-3, 3e-3]
EPOCHS_CHOICES = [50, 100]
BATCH = 64
L2 = 1e-4
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

class CNN:
    def __init__(self, H, C, seed):
        self.H = H
        self.C = C
        rng = np.random.default_rng(seed)
        self.W_conv = rng.normal(0, np.sqrt(2.0/(3*3)), (H, 1, 3, 3)).astype(np.float32)
        self.b_conv = np.zeros(H, np.float32)
        fan_in = H * 4 * 4
        self.W_fc = rng.normal(0, np.sqrt(2.0/fan_in), (fan_in, C)).astype(np.float32)
        self.b_fc = np.zeros(C, np.float32)
    
    def forward(self, x):
        N = x.shape[0]
        out_conv = np.zeros((N, self.H, 8, 8), np.float32)
        for n in range(N):
            for h in range(self.H):
                for i in range(8):
                    for j in range(8):
                        val = self.b_conv[h]
                        for di in range(3):
                            for dj in range(3):
                                ii = i + di - 1
                                jj = j + dj - 1
                                if 0 <= ii < 8 and 0 <= jj < 8:
                                    val += x[n, 0, ii, jj] * self.W_conv[h, 0, di, dj]
                        out_conv[n, h, i, j] = val
        out_relu = np.maximum(out_conv, 0)
        out_pool = np.zeros((N, self.H, 4, 4), np.float32)
        for n in range(N):
            for h in range(self.H):
                for i in range(4):
                    for j in range(4):
                        patch = out_relu[n, h, 2*i:2*i+2, 2*j:2*j+2]
                        out_pool[n, h, i, j] = patch.max()
        flat = out_pool.reshape(N, -1)
        logits = flat @ self.W_fc + self.b_fc
        return logits, (out_conv, out_relu, out_pool, flat)
    
    def loss_and_grad(self, x, y, l2):
        logits, cache = self.forward(x)
        N = x.shape[0]
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        dlogits = probs
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N
        
        out_conv, out_relu, out_pool, flat = cache
        
        dW_fc = flat.T @ dlogits + l2 * self.W_fc
        db_fc = dlogits.sum(axis=0)
        
        dflat = dlogits @ self.W_fc.T
        dpool = dflat.reshape(N, self.H, 4, 4)
        
        drelu = np.zeros_like(out_relu)
        for n in range(N):
            for h in range(self.H):
                for i in range(4):
                    for j in range(4):
                        patch = out_relu[n, h, 2*i:2*i+2, 2*j:2*j+2]
                        max_val = patch.max()
                        for di in range(2):
                            for dj in range(2):
                                if patch[di, dj] == max_val:
                                    drelu[n, h, 2*i+di, 2*j+dj] += dpool[n, h, i, j]
        
        dconv = drelu * (out_conv > 0)
        
        dW_conv = np.zeros_like(self.W_conv)
        db_conv = np.zeros_like(self.b_conv)
        for n in range(N):
            for h in range(self.H):
                for i in range(8):
                    for j in range(8):
                        grad = dconv[n, h, i, j]
                        db_conv[h] += grad
                        for di in range(3):
                            for dj in range(3):
                                ii = i + di - 1
                                jj = j + dj - 1
                                if 0 <= ii < 8 and 0 <= jj < 8:
                                    dW_conv[h, 0, di, dj] += x[n, 0, ii, jj] * grad
        dW_conv += l2 * self.W_conv
        
        loss += 0.5 * l2 * (np.sum(self.W_conv**2) + np.sum(self.W_fc**2))
        return loss, (dW_conv, db_conv, dW_fc, db_fc)

def train_model(X_train, y_train, X_val, y_val, H, lr, epochs, seed):
    C = 10
    model = CNN(H, C, seed)
    params = [model.W_conv, model.b_conv, model.W_fc, model.b_fc]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X_train.shape[0]
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_train[idx], y_train[idx]
            loss, grads = model.loss_and_grad(xb, yb, L2)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                ms[i] = beta1 * ms[i] + (1 - beta1) * g
                vs[i] = beta2 * vs[i] + (1 - beta2) * (g * g)
                m_hat = ms[i] / (1 - beta1 ** t)
                v_hat = vs[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    logits, _ = model.forward(X_val)
    preds = logits.argmax(axis=1)
    acc = (preds == y_val).mean()
    return model, acc

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    X = X.astype(np.float32) / 16.0
    X = X.reshape(-1, 1, 8, 8)
    
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    best_acc = -1
    best_config = None
    best_model = None
    
    for H in H_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCHS_CHOICES:
                model, acc = train_model(X_train, y_train, X_val, y_val, H, lr, epochs, SEED)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (H, lr, epochs)
                    best_model = model
    print(f"Best config: H={best_config[0]}, lr={best_config[1]}, epochs={best_config[2]}, val_acc={best_acc:.4f}")
    
    full_model, _ = train_model(X, y, X, y, *best_config, SEED)
    
    H, C = full_model.H, full_model.C
    magic = 0x434E4E31
    d = 64
    W_conv_flat = full_model.W_conv.reshape(-1)
    b_conv_flat = full_model.b_conv
    W_fc_flat = full_model.W_fc.reshape(-1)
    b_fc_flat = full_model.b_fc
    
    expected_floats = H*9 + H + H*16*C + C
    all_weights = np.concatenate([W_conv_flat, b_conv_flat, W_fc_flat, b_fc_flat])
    assert all_weights.shape[0] == expected_floats
    
    header = struct.pack("<4i", magic, d, H, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(all_weights.astype("<f4").tobytes())

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
    if (nw < 16 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || C != 10 || H <= 0 || H > 32) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected_floats = (long)H * 9 + H + (long)H * 16 * C + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *weights = (const float *)(hd + 4);
    const float *W_conv = weights;
    const float *b_conv = W_conv + (long)H * 9;
    const float *W_fc = b_conv + H;
    const float *b_fc = W_fc + (long)H * 16 * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(sizeof(float) * H * 8 * 8);
    float *relu_out = malloc(sizeof(float) * H * 8 * 8);
    float *pool_out = malloc(sizeof(float) * H * 4 * 4);
    float *fc_in = malloc(sizeof(float) * H * 16);
    float *logits = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        
        // Normalize input to [0,1]
        float input[64];
        for (int p = 0; p < 64; p++) input[p] = xi[p] / 16.0f;
        
        // Conv + bias
        for (int h = 0; h < H; h++) {
            for (int i2 = 0; i2 < 8; i2++) {
                for (int j2 = 0; j2 < 8; j2++) {
                    float val = b_conv[h];
                    for (int di = 0; di < 3; di++) {
                        for (int dj = 0; dj < 3; dj++) {
                            int ii = i2 + di - 1;
                            int jj = j2 + dj - 1;
                            if (ii >= 0 && ii < 8 && jj >= 0 && jj < 8) {
                                val += input[ii * 8 + jj] * W_conv[((h * 3 + di) * 3 + dj)];
                            }
                        }
                    }
                    conv_out[(h * 8 + i2) * 8 + j2] = val;
                }
            }
        }
        
        // ReLU
        for (int idx = 0; idx < H * 64; idx++) {
            relu_out[idx] = conv_out[idx] > 0 ? conv_out[idx] : 0.0f;
        }
        
        // MaxPool 2x2
        for (int h = 0; h < H; h++) {
            for (int i2 = 0; i2 < 4; i2++) {
                for (int j2 = 0; j2 < 4; j2++) {
                    float maxv = -1e30f;
                    for (int di = 0; di < 2; di++) {
                        for (int dj = 0; dj < 2; dj++) {
                            float v = relu_out[((h * 8 + i2*2 + di) * 8 + j2*2 + dj)];
                            if (v > maxv) maxv = v;
                        }
                    }
                    pool_out[(h * 4 + i2) * 4 + j2] = maxv;
                }
            }
        }
        
        // Flatten
        for (int idx = 0; idx < H * 16; idx++) {
            fc_in[idx] = pool_out[idx];
        }
        
        // FC
        for (int c = 0; c < C; c++) {
            float sum = b_fc[c];
            for (int idx = 0; idx < H * 16; idx++) {
                sum += fc_in[idx] * W_fc[idx * C + c];
            }
            logits[c] = sum;
        }
        
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```