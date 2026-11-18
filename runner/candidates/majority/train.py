"""Majority baseline for any label task: weights.bin is the most frequent training key (ties: the smallest bytes).
python3 -s -P train.py card.json train.in train.key weights.bin"""
import collections, struct, sys

data = open(sys.argv[3], "rb").read()
keys, i = [], 0
while i < len(data):
    k = struct.unpack_from("<I", data, i)[0]
    keys.append(data[i + 4:i + 4 + k])
    i += 4 + k
cnt = collections.Counter(keys)
best = min(cnt, key=lambda r: (-cnt[r], r))
open(sys.argv[4], "wb").write(best)
