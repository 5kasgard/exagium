# Task: make a stuck OpenTimestamps proof visible

You are working in `record/` of this repository, which is in /workspace (`record/record.py` and its tests in `record/test_record.py`).

**Background:** a scheduled job that completes pending OpenTimestamps proofs failed silently for five days. Nothing noticed, because `check-anchors` treats a pending proof (one not yet confirmed in Bitcoin) as acceptable.

**Required:** add an opt-in option `--max-pending-hours H` to `python3 record/record.py check-anchors`.
- Without the option, behaviour is unchanged: pending proofs are still accepted (exit 0).
- With the option, if any range head's proof is still pending and its Sigstore attestation time is more than H hours ago, exit 1. Print to stderr a message containing `still pending N h`, where N is the whole number of hours since the attestation (for example `still pending 72 h`).
- A pending proof younger than H hours is fine (exit 0).
- All existing tests in `record/test_record.py` must still pass.

Run the tests with `cd record && python3 -m unittest test_record`.
