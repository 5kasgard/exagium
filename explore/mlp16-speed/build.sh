#!/bin/sh
# Build every program with the project's flags (loop.py: -O2 -std=c11 -ffp-contract=off, no -march).
# Usage: sh build.sh   (run from this folder). Output in bin/. Add your own variants to src/ and to this file.
set -e
L=../../loop
F="-O2 -std=c11 -ffp-contract=off"
mkdir -p bin
gcc $F -o bin/linear $L/baselines/linear/infer.c
gcc $F -o bin/thermo $L/baselines/thermo/infer.c
for h in 16 32 64; do gcc $F -o bin/mlp$h $L/candidates/mlp$h/infer.c; done
for nb in 1 2 4 8; do for bl in 0 1; do gcc $F -DNB=$nb -DBRANCHLESS=$bl -o bin/mlp16_nb${nb}_bl$bl src/mlp_interleave.c; done; done
gcc --version | head -1
echo "built: $(ls bin | tr '\n' ' ')"
