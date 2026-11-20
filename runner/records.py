"""Record files: each record is a 4-byte little-endian length followed by that many bytes (runner/CONTRACT.md)."""
import json
import re
import struct

_SECRET_NAME = re.compile(r"(?i)(?:key|token|secret|password)")
_API_KEY = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:"
    r"sk[-_][a-z0-9_-]*"
    r"|pk[-_][a-z0-9_-]*"
    r"|rk[-_][a-z0-9_-]*"
    r"|gh[pousr]_[a-z0-9]*"
    r"|akia[0-9a-z]*"
    r"|asia[0-9a-z]*"
    r"|aiza[0-9a-z_-]*"
    r"|xox[baprs]-[a-z0-9-]*"
    r"|hf_[a-z0-9]*"
    r"|gsk_[a-z0-9]*"
    r"|r8_[a-z0-9]*"
    r")"
)


def _scrub(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _SECRET_NAME.search(k):
                continue
            out[k] = _scrub(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and _API_KEY.search(value):
        raise ValueError("record contains a value that looks like an API key")
    return value


def _record_bytes(r):
    if r is None or isinstance(r, (dict, list, tuple, str, int, float, bool)):
        return json.dumps(_scrub(r), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return bytes(r)


def pack(recs):
    return b"".join(struct.pack("<I", len(r)) + r for r in (_record_bytes(x) for x in recs))


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
