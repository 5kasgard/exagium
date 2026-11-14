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
gcc $F -o bin/clock src/clock.c
gcc $F -o bin/h1_const src/h1_const.c
gcc $F -o bin/h1_half src/h1_half.c
gcc $F -o bin/h2_pre src/h2_pre.c
gcc $F -o bin/h3_l1only src/h3_l1only.c
gcc $F -o bin/h3_noarg src/h3_noarg.c
gcc $F -o bin/h4_noh src/h4_noh.c
gcc $F -o bin/h5_u1 src/h5_u1.c
gcc $F -o bin/h9_l2bl src/h9_l2bl.c
gcc $F -o bin/h9_bl src/h9_bl.c
gcc $F -o bin/ceil src/ceil.c
# add your own variants above this line
gcc --version | head -1
echo "built: $(ls bin | tr '\n' ' ')"
