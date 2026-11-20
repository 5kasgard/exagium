"""Record files: each record is a 4-byte little-endian length followed by that many bytes (runner/CONTRACT.md)."""
import json
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


def derive_outcome_stats(path):
    """
    Read call records from path and return per-(pool, model) stats.

    Returns a tuple (stats, skipped) where:
      stats: dict mapping (pool, model) -> {
          "calls": int,           # total calls for this pool/model
          "truncated": int,       # calls where finish_reason/stop_reason indicates output limit
          "max_output_tokens": int,  # max output_tokens among non-truncated calls (0 if none)
      }
      skipped: int                # records missing required fields or failing to parse
    """
    records = read(path)
    stats = {}
    skipped = 0

    for rec in records:
        try:
            data = json.loads(rec.decode("utf-8"))
        except Exception:
            skipped += 1
            continue

        # Required fields
        pool = data.get("pool")
        model = data.get("model")
        finish_reason = data.get("finish_reason") or data.get("stop_reason")
        output_tokens = data.get("output_tokens")

        if pool is None or model is None or finish_reason is None or output_tokens is None:
            skipped += 1
            continue

        key = (pool, model)
        if key not in stats:
            stats[key] = {"calls": 0, "truncated": 0, "max_output_tokens": 0}

        stats[key]["calls"] += 1

        # Check if truncated: finish_reason/stop_reason == "length" typically means hit output token limit
        is_truncated = (finish_reason == "length")
        if is_truncated:
            stats[key]["truncated"] += 1
        else:
            # Track max output tokens for non-truncated calls
            if output_tokens > stats[key]["max_output_tokens"]:
                stats[key]["max_output_tokens"] = output_tokens

    return stats, skipped
