#!/usr/bin/env python3
"""Build the input files from the committed training file (never the test file).
dev.bin = rows 2867.. of loop/task/optdigits.tra (956 images x 64 bytes); mid.bin = dev x 10; big.bin = dev x 100.
Usage: python3 make_data.py   (run from this folder)"""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'loop', 'task', 'optdigits.tra')
rows = [list(map(int, l.split(','))) for l in open(src) if l.strip()]
dev = rows[len(rows) * 3 // 4:]
raw = bytes(v for r in dev for v in r[:64])
assert len(raw) == 956 * 64 and all(b <= 16 for b in raw)
os.makedirs(os.path.join(here, 'data'), exist_ok=True)
for name, k in (('dev.bin', 1), ('mid.bin', 10), ('big.bin', 100)):
    open(os.path.join(here, 'data', name), 'wb').write(raw * k)
print('wrote data/dev.bin (956), data/mid.bin (9,560), data/big.bin (95,600 images)')
