"""All-false baseline for sat: nothing to learn; writes an empty weights.bin.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import sys

open(sys.argv[4], "wb").close()
