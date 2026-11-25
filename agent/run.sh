#!/usr/bin/env bash
# Runs opencode for one /oc request (.github/workflows/opencode.yml). The calling step holds at most one provider's key.
# MODE, MODEL, PROVIDER and OUT come from the step's env; agent/oc_pick.py has already checked them.
set -u
set +x   # never trace: the step's env holds a key
case "$MODE" in
  ask)
    timeout 900 agent/node_modules/.bin/opencode run --agent answer -m "$MODEL" -f "$OUT/prompt.md" \
      "Answer the request in the attached file. Read the repository files you need and name the paths you used." \
      > "$OUT/answer.md" 2> "$RUNNER_TEMP/err.txt" || echo "The run failed (exit $?); see the workflow log." >> "$OUT/answer.md" ;;
  models)
    timeout 120 agent/node_modules/.bin/opencode models "$PROVIDER" > "$OUT/answer.md" 2> "$RUNNER_TEMP/err.txt" \
      || echo "Listing failed; see the workflow log." >> "$OUT/answer.md" ;;
  *) echo "unknown mode" > "$OUT/answer.md" ;;
esac
python3 -I agent/scrub.py "$OUT/answer.md"
