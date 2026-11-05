#!/bin/sh
# Hypothesis 1 single-host checks, run before the counter had its sandbox. Uses cg.py (raw cachegrind,
# caller's environment) so each factor can be switched on and off. Output: h1_local.txt
# usage: sh run.sh <path to counter/workloads> > h1_local.txt
W=$1
gcc -O2 -o /tmp/h1mlp "$W/mlp.c"
for m in pin pin_noaslr; do echo "== C mlp 1000, $m"; python3 cg.py 5 $m /tmp/h1mlp 1000; done
for m in pin pin_hashseed pin_noaslr_hashseed; do echo "== pure Python mlp.py 20, $m"; python3 cg.py 4 $m python3 "$W/mlp.py" 20; done
for m in pin_hashseed pin_hashseed_1thr pin_hashseed_1thr_noaslr; do echo "== numpy mlp_np.py 256, $m"; python3 cg.py 4 $m python3 mlp_np.py 256; done
echo "== working-directory length, pure Python mlp.py 3 (one rep each)"
for d in /tmp/aa /tmp/aaaaaaaaaaaaaaaaaaaaaa; do mkdir -p $d; cp "$W/mlp.py" $d/; (cd $d && echo "cwd $d" && python3 "$OLDPWD/cg.py" 1 pin_hashseed python3 mlp.py 3 | grep "^Ir"); done
echo "== HOME length, C mlp 1000 (one rep each)"
for h in /root /a/much/longer/home/directory/than/before; do echo "HOME $h"; HOME=$h python3 cg.py 1 pin /tmp/h1mlp 1000 | grep -E "^(Ir|D1mr)"; done
