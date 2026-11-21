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


def _iter_calls(recs):
    if isinstance(recs, (str, bytes)):
        with open(recs, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        for rec in recs:
            if isinstance(rec, (str, bytes)):
                rec = json.loads(rec)
            yield rec


def call_stats(recs):
    """Per (pool, model) call counts, truncations and max safe output tokens from JSON call records.

    A record needs pool/engine, model, an output token count (output_tokens, completion_tokens or
    usage.completion_tokens) and a stop reason (finish, finish_reason or stop_reason). Records missing
    any of these, or with an invalid token count, are skipped and counted separately.
    Returns ((pool, model) -> {"calls", "truncated", "max_output_tokens"}, skipped).
    """
    stats, skipped = {}, 0
    trunc_words = ("length", "max_tokens", "token_limit", "max_output", "limit_reached")
    for rec in _iter_calls(recs):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        pool = rec.get("pool", rec.get("engine"))
        model = rec.get("model")
        reason = rec.get("finish", rec.get("finish_reason", rec.get("stop_reason")))
        tokens = rec.get("output_tokens", rec.get("completion_tokens"))
        if tokens is None and isinstance(rec.get("usage"), dict):
            usage = rec["usage"]
            tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if pool is None or model is None or reason is None or tokens is None:
            skipped += 1
            continue
        try:
            tokens = int(tokens)
        except (TypeError, ValueError):
            skipped += 1
            continue
        s = stats.setdefault((pool, model), {"calls": 0, "truncated": 0, "max_output_tokens": 0})
        s["calls"] += 1
        reason_s = str(reason).lower()
        truncated = any(w in reason_s for w in trunc_words)
        if "max_tokens" in rec:
            try:
                truncated = truncated or tokens >= int(rec["max_tokens"])
            except (TypeError, ValueError):
                pass
        if truncated:
            s["truncated"] += 1
        else:
            s["max_output_tokens"] = max(s["max_output_tokens"], tokens)
    return stats, skipped
