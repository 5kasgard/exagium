"""WalkSAT: nothing is learned; weights.bin holds its fixed settings (uint32 noise in 1/1000, uint32 flips per instance).
python3 -s -P train.py card.json train.in train.key weights.bin"""
import struct, sys

open(sys.argv[4], "wb").write(struct.pack("<II", 500, 200000))
