#!/usr/bin/env python3
"""Demonstrate that Python's json.dumps(sort_keys=True) is NOT a cross-language
canonical JSON form, and that RFC 8785 JCS produces a different digest for the
same logical record.

The exagium RESULTS.md mentions: 'Python's json.dumps(sort_keys) isn't a
cross-language canonical form; RFC 8785 JCS is the candidate.' This script
demonstrates the divergence on the exagium sample record itself, plus on a
minimal synthetic example showing where the two diverge.
"""
import json, hashlib, sys

# --- real record ---
SAMPLE = "/home/z/my-project/upload/exagium-anchors/local/sample_record.json"
with open(SAMPLE) as f:
    record = json.load(f)

# Python's "canonical" form
py_canon = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
py_digest = hashlib.sha256(py_canon.encode('utf-8')).hexdigest()

# RFC 8785 JCS — apply the canonicalization rules manually (no Python lib exists
# in this sandbox that does JCS; we implement the rules directly).
# Rules (https://www.rfc-editor.org/rfc/rfc8785.txt):
#   1. Sort object keys by UTF-16 code unit (NOT Python's UTF-8 code point).
#      For BMP characters this matches; for chars outside BMP (emoji etc.) it differs.
#   2. Strip insignificant whitespace.
#   3. Numbers: serialize the shortest form that round-trips to the same float.
#      Python's repr() does this for floats; ints are simple.
#   4. Strings: escape per RFC 8252 §7 with `\"` `\\` and codepoints < 0x20 as \u00XX.
def jcs_serialize(obj):
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        # RFC 8785 §3.2.2: use the shortest round-tripping form.
        # Python's repr() does this for floats since 3.1.
        if obj != obj:  # NaN
            return "NaN"
        if obj == float('inf'):
            return "Infinity"
        if obj == float('-inf'):
            return "-Infinity"
        # Use repr() and lowercase e
        s = repr(obj)
        if 'e' in s:
            s = s.replace('e', 'E')
        return s
    if isinstance(obj, str):
        out = ['"']
        for ch in obj:
            cp = ord(ch)
            if ch == '"':
                out.append('\\"')
            elif ch == '\\':
                out.append('\\\\')
            elif cp < 0x20:
                out.append(f'\\u{cp:04x}')
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)
    if isinstance(obj, list):
        return "[" + ",".join(jcs_serialize(x) for x in obj) + "]"
    if isinstance(obj, dict):
        # RFC 8785 §3.2.3: sort by UTF-16 code units of the encoded key
        # For BMP characters, UTF-16 code unit order = UTF-8 byte order.
        # For astral characters (cp > 0xFFFF), Python str is a single codepoint
        # but UTF-16 represents them as surrogate pairs (2 code units).
        # We must encode each key to UTF-16-LE bytes and sort by that.
        def utf16_key(k):
            return k.encode('utf-16-le')
        items = sorted(obj.items(), key=lambda kv: utf16_key(kv[0]))
        return "{" + ",".join(f'{jcs_serialize(k)}:{jcs_serialize(v)}' for k, v in items) + "}"
    raise TypeError(f"unsupported type {type(obj)}")

jcs_canon = jcs_serialize(record)
jcs_digest = hashlib.sha256(jcs_canon.encode('utf-8')).hexdigest()

print(f"sample record: {SAMPLE}")
print(f"  Python json.dumps(sort_keys=True) digest: {py_digest}")
print(f"  RFC 8785 JCS digest:                      {jcs_digest}")
print(f"  digests match: {py_digest == jcs_digest}")
print()
# Show the bytes-level difference
py_bytes = py_canon.encode('utf-8')
jcs_bytes = jcs_canon.encode('utf-8')
print(f"  python canonical bytes: {len(py_bytes)}")
print(f"  jcs canonical bytes:    {len(jcs_bytes)}")
# Find the first divergence
for i, (a, b) in enumerate(zip(py_bytes, jcs_bytes)):
    if a != b:
        print(f"  first byte difference at offset {i}:")
        print(f"    python: ...{py_bytes[max(0,i-20):i+30].decode('utf-8', 'replace')!r}...")
        print(f"    jcs:    ...{jcs_bytes[max(0,i-20):i+30].decode('utf-8', 'replace')!r}...")
        break
else:
    if len(py_bytes) != len(jcs_bytes):
        print(f"  same prefix but different length ({len(py_bytes)} vs {len(jcs_bytes)})")
print()

# --- synthetic divergence example ---
print("--- synthetic example showing where they diverge ---")
example = {
    "emoji_key": "value",  # pure-ASCII key
    "result": 0.1 + 0.2,   # 0.30000000000000004
    "name": "café",
}
py = json.dumps(example, sort_keys=True, separators=(',', ':'))
jcs = jcs_serialize(example)
print(f"  python: {py}")
print(f"  jcs:    {jcs}")
print(f"  python sha256: {hashlib.sha256(py.encode()).hexdigest()}")
print(f"  jcs sha256:    {hashlib.sha256(jcs.encode()).hexdigest()}")
print()
print("=> A Go or Rust verifier that uses its standard JSON encoder + sort_keys")
print("   will NOT produce the same digest as Python's json.dumps(sort_keys=True).")
print("   RFC 8785 JCS is the only widely-agreed cross-language canonical form.")
print("   Recommendation: exagium should hash the *stored bytes* of the record (the")
print("   UTF-8 bytes committed to git), OR adopt RFC 8785 JCS via a vetted library")
print("   in every language that consumes the digest.")
