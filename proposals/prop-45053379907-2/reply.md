```hypothesis
We propose a "label-tree" family: a context tree (trie) storing the majority next character for every suffix of the 16-byte training contexts. At inference, it walks the tree from the most recent byte backward, using the longest matching suffix. This variable-order Markov model captures non-linear, context-specific patterns in character sequences more effectively than a linear model, while remaining extremely fast (single tree walk per item). If dev accuracy does not exceed label-linear (0.3375) or inference cost exceeds label-linear's ~7k Ir/item, the hypothesis fails.
```

```python
import json, struct, sys
from collections import deque

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = records(sys.argv[2])
    y = [r[0] for r in records(sys.argv[3])]
    n = len(X)
    assert all(len(x) == d for x in X)
    assert all(0 <= c < C for c in y)

    # Tree nodes: each node has counts[C] and children dict byte->node_index
    nodes_counts = []
    nodes_children = []

    def new_node():
        nodes_counts.append([0]*C)
        nodes_children.append({})
        return len(nodes_counts)-1

    root = new_node()

    for ctx_bytes, target in zip(X, y):
        node = root
        nodes_counts[node][target] += 1
        for i in range(d-1, -1, -1):
            b = ctx_bytes[i]
            children = nodes_children[node]
            if b in children:
                node = children[b]
            else:
                child = new_node()
                children[b] = child
                node = child
            nodes_counts[node][target] += 1

    # Majority prediction per node (tie -> smallest class)
    predictions = []
    for counts in nodes_counts:
        best = 0
        max_cnt = -1
        for c, cnt in enumerate(counts):
            if cnt > max_cnt:
                max_cnt = cnt
                best = c
        predictions.append(best)

    # BFS to assign deterministic indices
    order = []
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for b in sorted(nodes_children[u].keys()):
            q.append(nodes_children[u][b])

    old_to_new = {old: new for new, old in enumerate(order)}
    num_nodes = len(order)

    new_predictions = [0]*num_nodes
    new_children = [{} for _ in range(num_nodes)]
    for new_idx, old_idx in enumerate(order):
        new_predictions[new_idx] = predictions[old_idx]
        for b, old_child in nodes_children[old_idx].items():
            new_children[new_idx][b] = old_to_new[old_child]

    # Flatten edges
    edges = []
    node_child_start = [0]*num_nodes
    node_child_count = [0]*num_nodes
    for i in range(num_nodes):
        node_child_start[i] = len(edges)
        for b in sorted(new_children[i].keys()):
            edges.append((b, new_children[i][b]))
        node_child_count[i] = len(edges) - node_child_start[i]
    num_edges = len(edges)

    # Write weights.bin
    magic = 0x54524545  # "TREE"
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<IIIII", magic, d, C, num_nodes, num_edges))
        for i in range(num_nodes):
            pred = new_predictions[i]
            cnt = node_child_count[i]
            start = node_child_start[i]
            f.write(struct.pack("<BHI", pred, cnt, start))
            f.write(b'\x00')  # padding to 8 bytes
        for b, child in edges:
            f.write(struct.pack("<BI", b, child))

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }

    uint32_t magic = w[0] | w[1]<<8 | w[2]<<16 | (uint32_t)w[3]<<24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t d = w[4] | w[5]<<8 | w[6]<<16 | (uint32_t)w[7]<<24;
    uint32_t C = w[8] | w[9]<<8 | w[10]<<16 | (uint32_t)w[11]<<24;
    uint32_t num_nodes = w[12] | w[13]<<8 | w[14]<<16 | (uint32_t)w[15]<<24;
    uint32_t num_edges = w[16] | w[17]<<8 | w[18]<<16 | (uint32_t)w[19]<<24;

    size_t expected = 20 + (size_t)num_nodes * 8 + (size_t)num_edges * 5;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *nodes_buf = w + 20;
    unsigned char *edges_buf = nodes_buf + (size_t)num_nodes * 8;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }

        uint32_t node = 0;
        unsigned char pred = nodes_buf[0];  // root prediction at offset 0

        for (int j = d - 1; j >= 0; j--) {
            unsigned char b = xi[j];
            uint16_t cnt = nodes_buf[node*8 + 1] | (nodes_buf[node*8 + 2] << 8);
            uint32_t start = nodes_buf[node*8 + 3] | (nodes_buf[node*8 + 4] << 8) |
                             (nodes_buf[node*8 + 5] << 16) | ((uint32_t)nodes_buf[node*8 + 6] << 24);
            unsigned char *edge_ptr = edges_buf + start * 5;
            int found = 0;
            for (uint16_t e = 0; e < cnt; e++) {
                if (edge_ptr[0] == b) {
                    uint32_t child = edge_ptr[1] | (edge_ptr[2] << 8) | (edge_ptr[3] << 16) | ((uint32_t)edge_ptr[4] << 24);
                    node = child;
                    pred = nodes_buf[node*8];
                    found = 1;
                    break;
                }
                edge_ptr += 5;
            }
            if (!found) break;
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```