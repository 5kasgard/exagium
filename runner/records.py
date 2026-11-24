"""Record files: each record is a 4-byte little-endian length followed by that many bytes (runner/CONTRACT.md)."""
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
    """Standard error of the mean with the variance summed within each writer cluster.

    Uses the usual cluster-robust small-sample factors G/(G-1) and (n-1)/(n-k), with k = 1 for a mean.
    `writers` is the writer id of each item; a None writer is a caller bug and raises ValueError.
    Returns None when fewer than two writers are present (the factor G/(G-1) is undefined)."""
    n = len(values)
    if len(writers) != n:
        raise ValueError("values and writers must have the same length")
    if any(w is None for w in writers):
        raise ValueError("writer id is None; the caller must drop those items first")
    if n < 2:
        return None
    clusters = {}
    for v, w in zip(values, writers):
        clusters.setdefault(w, []).append(v)
    G = len(clusters)
    if G < 2:
        return None
    mean = sum(values) / n
    sumsq = sum((sum(v - mean for v in vs)) ** 2 for vs in clusters.values())
    var = (G / (G - 1)) * ((n - 1) / (n - 1)) * sumsq / n ** 2
    return var ** 0.5
