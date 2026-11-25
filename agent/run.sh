#!/usr/bin/env bash
# Runs opencode for one /oc request (.github/workflows/opencode.yml). The calling step holds at most one provider's key.
# MODE, MODEL, PROVIDER and OUT come from the step's env; agent/oc_pick.py has already checked them.
set -u
set +x   # never trace: the step's env holds a key
case "$MODE" in
  ask)
    # the request goes in as the message, after "--": -f is an array option and took the message as a second file
    # (run 45519136124: "File not found: Answer the request ...")
    msg="$(printf 'Answer the request below. Read the repository files you need and name the paths you used.\n\n%s' "$(cat "$OUT/prompt.md")")"
    timeout 900 agent/node_modules/.bin/opencode run --agent answer -m "$MODEL" -- "$msg" \
      > "$OUT/answer.md" 2> "$RUNNER_TEMP/err.txt" || echo "The run failed (exit $?); see the workflow log." >> "$OUT/answer.md" ;;
  models)
    timeout 120 agent/node_modules/.bin/opencode models "$PROVIDER" > "$OUT/answer.md" 2> "$RUNNER_TEMP/err.txt" \
      || echo "Listing failed; see the workflow log." >> "$OUT/answer.md" ;;
  *) echo "unknown mode" > "$OUT/answer.md" ;;
esac
python3 -I agent/scrub.py "$OUT/answer.md"
