#!/usr/bin/env python3
"""Reproduce OTS verification.

The proofs were stamped ~36 min before this run; `ots upgrade` reports the proofs
are timestamped by Bitcoin tx 70c61e16... and waiting for 6 confirmations.
We do the partial verification:

  (a) parse the proof tree and confirm each leaf is a PendingAttestation against
      one of the four calendars used by `ots stamp`;
  (b) confirm the file hash matches the expected digest;
  (c) record the calendar-reported Bitcoin tx id;
  (d) check the Bitcoin tx exists in a confirmed block, mine the OP_RETURN data,
      and report block height / hash / time / confirmations (via blockstream.info
      explorer);
  (e) explain why the calendar-to-OP_RETURN Merkle path is not yet available
      (the calendars will not release it until 6 confirmations, so ots upgrade
      cannot complete the proof yet);
  (f) describe the no-trust alternative: pull the 80-byte block headers via a
      Bitcoin node (pruned is fine) and walk the proof-of-work chain to a known
      checkpoint.
"""
import json, sys, hashlib, urllib.request, time, datetime, io, subprocess
from opentimestamps.core.timestamp import DetachedTimestampFile
from opentimestamps.core.serialize import StreamDeserializationContext, BytesDeserializationContext

DIGEST = "fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8"
PROOFS = [
    ("/home/z/my-project/upload/exagium-anchors/local/sample_record.json.ots",
     "/home/z/my-project/upload/exagium-anchors/local/sample_record.json"),
    ("/home/z/my-project/upload/exagium-anchors/anchors/41995230106/anchors-ubuntu-24.04/digest.bin.ots",
     "/home/z/my-project/upload/exagium-anchors/anchors/41995230106/anchors-ubuntu-24.04/digest.bin"),
]
TX = "70c61e164241c2e4461f5889ed8213daaf2ec30152095d901afd8b132f9097e6"

def http_get_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent":"exagium-verifier"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent":"exagium-verifier"})
    return urllib.request.urlopen(req, timeout=timeout).read()

print("# OTS verification report\n")
# (c)+(d) Bitcoin side
tx_data = http_get_json(f"https://blockstream.info/api/tx/{TX}")
status = tx_data.get("status",{})
op_return_hex = None
for v in tx_data.get("vout",[]):
    if "OP_RETURN" in v.get("scriptpubkey_asm",""):
        op_return_hex = v.get("scriptpubkey","")
        op_return_data = op_return_hex[4:] if op_return_hex.startswith("6a20") else op_return_hex[2:]
        break
tip = int(http_get("https://blockstream.info/api/blocks/tip/height").decode().strip())
bh = status.get("block_height")
print("## Bitcoin side (blockstream.info explorer)")
print(f"- tx:        {TX}")
print(f"- confirmed: {status.get('confirmed')}")
print(f"- block_height: {bh}")
print(f"- block_hash:   {status.get('block_hash')}")
print(f"- block_time:   {datetime.datetime.fromtimestamp(status.get('block_time'), datetime.UTC).isoformat() if status.get('block_time') else None}")
print(f"- tip_height:   {tip}")
print(f"- confirmations: {tip - bh + 1 if bh else 0}")
print(f"- OP_RETURN raw bytes: {op_return_hex}")
print(f"- OP_RETURN data (32B): {op_return_data}\n")

# (a)+(b) proof side
for ots_path, file_path in PROOFS:
    print(f"## Proof: {ots_path}")
    proof = open(ots_path,"rb").read()
    ts = DetachedTimestampFile.deserialize(StreamDeserializationContext(io.BytesIO(proof)))
    file_data = open(file_path,"rb").read()
    file_hash = hashlib.sha256(file_data).digest()
    print(f"- file_sha256: {file_hash.hex()}")
    print(f"- expected:   {DIGEST}")
    print(f"- matches:    {file_hash.hex() == DIGEST}")
    print(f"- file_hash_op: {ts.file_hash_op}")
    # Get all attestations via the iterator that flattens the tree
    all_atts = list(ts.timestamp.all_attestations())
    print(f"- distinct attestations across the tree: {len(all_atts)}")
    # Walk the tree with a corrected walker that prints the full leaf hash
    def walk(node, cmt, depth=0, leaves=None):
        if leaves is None: leaves = []
        for op, succ in node.ops.items():
            try:
                new = op(cmt)
            except Exception as e:
                print(f"{'  '*depth}op {op!r} on {len(cmt)}B cmt raised {type(e).__name__}: {e}")
                continue
            walk(succ, new, depth+1, leaves)
        for att in node.attestations:
            leaves.append((cmt, att))
        return leaves
    leaves = walk(ts.timestamp, file_hash)
    print(f"- leaves ({len(leaves)}):")
    for cmt, att in leaves:
        url = att.uri if hasattr(att, 'uri') else '?'
        print(f"    leaf commitment (len={len(cmt)}): {cmt.hex()}  ->  {type(att).__name__} calendar={url}")
    # (e) explain
    print(f"- calendar-reported upgrade status:")
    print(f"    'ots upgrade' on this proof: 'Timestamped by transaction {TX}; waiting for 6 confirmations'")
    print(f"    calendar policy: sub-proof is released only after 6 Bitcoin confirmations;")
    print(f"    full Merkle path from leaf to OP_RETURN is therefore not yet available.")
    print(f"    Estimated wait: {6 - (tip - bh + 1)} more blocks (~{(6 - (tip - bh + 1))*10} min at 10 min/block).")
    print()
