# label-mlp2l: two-hidden-layer ReLU MLP (from P5 proposal prop-44492370928-2)

Written by a free model through P5 (proposals branch, prop-44492370928-2/reply.md); train.py and infer.c are copied
unchanged from the extracted reply. Admitted to runner/candidates by Claude's check (queue P5a, 2026-11-20), as
runner/proposals/PLAN.md requires: train.py reads only its four arguments (card, train.in, train.key, output) and
writes only the weights; numpy only; seeded; encodes inputs from the card's types, so it runs on any label task.
infer.c reads only the weights file and the input records (stdio, stdlib, string). The proposer's own claim and
falsifier, kept as written: two layers should beat label-mlp16's 0.977 on digits dev at about twice its cost; if not,
depth does not help here. Its digits dev result (0.9864) is one of 100+ on one split: see queue P5b for confirmation.
