# Task: `check-anchors` misses a deleted OpenTimestamps proof

You are working in `record/` of this repository, which is in /workspace (`record/record.py` and its tests in `record/test_record.py`).

**Bug report from an outside verifier:** in a ledger whose entries are covered by Sigstore range attestations, they deleted the OpenTimestamps proof file of a range head (for example `anchors/000002.ots`, where `anchors/000002.sigstore.json` attests the range ending at entry 2). Afterwards `python3 record/record.py check-anchors` still exited 0, so the missing proof went unreported.

**Required behaviour:**
- `check-anchors` must exit 1 when the `.ots` proof for any attested range head is missing.
- It must print a message to stderr containing `NNNNNN.ots is missing` (for example `000002.ots is missing`).
- With the proof present, the existing behaviour must not change.
- All existing tests in `record/test_record.py` must still pass.

Run the tests with `cd record && python3 -m unittest test_record`.
