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
gcc $F -o bin/mlp16_l1only src/mlp16_l1only.c
gcc $F -o bin/mlp16_bl_tree src/mlp16_bl_tree.c
gcc $F -o bin/mlp16_argmax_bl src/mlp16_argmax_bl.c
gcc $F -o bin/mlp16_bl_single_argmax_bl src/mlp16_bl_single_argmax_bl.c
gcc $F -o bin/mlp16_null src/mlp16_null.c
gcc $F -o bin/mlp16_l1_fixrow_dep src/mlp16_l1_fixrow_dep.c
gcc $F -o bin/mlp16_l1_fixrow_indep src/mlp16_l1_fixrow_indep.c
gcc $F -o bin/mlp16_chunked src/mlp16_chunked.c
gcc $F -o bin/mlp16_null_chunked src/mlp16_null_chunked.c
gcc $F -o bin/linear_null src/linear_null.c
gcc $F -DSTRAT=1 -o bin/st_bigbuf src/stage_null.c
gcc $F -DSTRAT=3 -o bin/st_mmap src/stage_null.c
gcc $F -DSTRAT=4 -o bin/st_mmap_pop src/stage_null.c
gcc $F -DSTRAT=5 -o bin/st_huge src/stage_null.c
for c in 16 64 256 1024 4096 16384; do gcc $F -DSTRAT=2 -DCH=$c -o bin/st_chunk$c src/stage_null.c; gcc $F -DSTRAT=6 -DCH=$c -o bin/st_read$c src/stage_null.c 2>/dev/null; done
gcc $F -o bin/mlp16_mmap src/mlp16_mmap.c
gcc $F -o bin/lgn_orig src/lgn_orig.c
gcc $F -o bin/lgn_blockread src/lgn_blockread.c
# add your own variants above this line
gcc --version | head -1
echo "built: $(ls bin | tr '\n' ' ')"
