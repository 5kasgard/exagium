"""Verify an OpenTimestamps proof against Bitcoin block headers fetched from two explorers.
Usage: ots_check.py <file> <file.ots>
Replays every op from the file's sha256 to each Bitcoin attestation and compares the result with
the merkle root in the block header (bytes 36..68), as served by blockstream.info and mempool.space."""
import sys, hashlib, urllib.request, datetime
from opentimestamps.core.timestamp import DetachedTimestampFile
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
from opentimestamps.core.serialize import StreamDeserializationContext

def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.read().decode().strip()

data = open(sys.argv[1], "rb").read()
with open(sys.argv[2], "rb") as f:
    dt = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
assert dt.file_digest == hashlib.sha256(data).digest(), "proof is for a different file"
print("file sha256", dt.file_digest.hex())
ok = 0
for msg, att in dt.timestamp.all_attestations():
    if not isinstance(att, BitcoinBlockHeaderAttestation):
        continue
    h = att.height
    for base in ("https://blockstream.info/api", "https://mempool.space/api"):
        bh = get(f"{base}/block-height/{h}")
        hdr = bytes.fromhex(get(f"{base}/block/{bh}/header"))
        assert hashlib.sha256(hashlib.sha256(hdr).digest()).digest()[::-1].hex() == bh, "header/hash mismatch"
        t = int.from_bytes(hdr[68:72], "little")
        match = hdr[36:68] == msg
        ok += match
        print(f"block {h} via {base.split('/')[2]:18} merkle match={match} time={datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()}")
print("matches", ok)
sys.exit(0 if ok else 1)
