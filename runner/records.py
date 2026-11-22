"""Record files: each record is a 4-byte little-endian length followed by that many bytes (runner/CONTRACT.md)."""
import math
import statistics
import struct


def pack(recs):
    return b"".join(struct.pack("<I", len(r)) + bytes(r) for r in recs)


def write(path, recs):
    with open(path, "wb") as f:
        f.write(pack(recs))


def unpack(data, max_len=None, count=None):
    """Parse bytes into records. Raises ValueError on a truncated file, an over-long record or a wrong count."""
    out, i, n = [], 0, len(data)
    while i < n:
        if i + 4 > n:
            raise ValueError(f"truncated length at byte {i}")
        k = struct.unpack_from("<I", data, i)[0]
        i += 4
        if max_len is not None and k > max_len:
            raise ValueError(f"record {len(out)} is {k} bytes, over the bound {max_len}")
        if i + k > n:
            raise ValueError(f"record {len(out)} truncated")
        out.append(data[i:i + k])
        i += k
        if count is not None and len(out) > count:
            raise ValueError(f"more than {count} records")
    if count is not None and len(out) != count:
        raise ValueError(f"{len(out)} records, expected {count}")
    return out


def read(path, **kw):
    with open(path, "rb") as f:
        return unpack(f.read(), **kw)


def clustered_se(values, writers):
    """SE of the mean with writer-clustered variance (CR1, k = 1 for a mean).

    Variance is (G/(G-1)) * ((n-1)/(n-k)) * sum_g (sum_{i in g}(x_i - mean))^2 / n^2, the
    same mean-SE convention as probe.summarise(). Returns None when there are fewer than 2
    writers, and raises ValueError if a writer id is None (the caller must skip such items).
    """
    if len(values) != len(writers):
        raise ValueError("values and writers must have the same length")
    if any(w is None for w in writers):
        raise ValueError("a writer id is None")
    n = len(values)
    G = len(set(writers))
    if G < 2:
        return None
    m = statistics.fmean(values)
    sums = {}
    for v, w in zip(values, writers):
        sums[w] = sums.get(w, 0.0) + (v - m)
    meat = sum(s * s for s in sums.values())
    k = 1
    var = (G / (G - 1)) * ((n - 1) / (n - k)) * meat / (n * n)
    return math.sqrt(var)
