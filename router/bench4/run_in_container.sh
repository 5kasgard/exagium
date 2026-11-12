#!/bin/sh
# bench-4: run one harness on one task inside the container, in /workspace, at the harness's own defaults.
# usage: run_in_container.sh HARNESS TASK_FILE OUT_DIR
# Env: BENCH4_BASE (the meter, e.g. http://172.17.0.1:8088), BENCH4_MODEL (OpenRouter slug).
# The key every harness sees is "dummy": the meter adds the real one on the host.
# Each harness gets the same task text, its own default system prompt and tools, no web tool, and no
# step limit of its own beyond a high one (the meter's call cap is the one budget for all).
set -u
H=$1; TASK=$2; OUT=$3
mkdir -p "$OUT"
export HOME=/root OPENROUTER_API_KEY=dummy
cd /workspace
case "$H" in
  goose)
    GOOSE_PROVIDER=openrouter GOOSE_MODEL="$BENCH4_MODEL" OPENROUTER_HOST="$BENCH4_BASE" GOOSE_MODE=auto \
    GOOSE_DISABLE_KEYRING=1 GOOSE_TELEMETRY_ENABLED=false \
      goose run --no-session --no-profile --with-builtin developer --max-turns 1000 -q --output-format json -i "$TASK" \
      > "$OUT/harness.out" 2> "$OUT/harness.err"; rc=$? ;;
  pi)
    mkdir -p /root/.pi/agent
    printf '{"providers":{"openrouter":{"baseUrl":"%s/api/v1"}}}' "$BENCH4_BASE" > /root/.pi/agent/models.json
    pi -p --no-session --mode json --provider openrouter --model "$BENCH4_MODEL" "$(cat "$TASK")" \
      > "$OUT/harness.out" 2> "$OUT/harness.err"; rc=$? ;;
  opencode)
    mkdir -p /root/.config/opencode
    cat > /root/.config/opencode/opencode.json <<EOF
{"\$schema":"https://opencode.ai/config.json","autoupdate":false,"share":"disabled",
 "model":"openrouter/$BENCH4_MODEL","small_model":"openrouter/$BENCH4_MODEL",
 "provider":{"openrouter":{"options":{"baseURL":"$BENCH4_BASE/api/v1","apiKey":"dummy"},
   "models":{"$BENCH4_MODEL":{"name":"bench4","tool_call":true}}}},
 "permission":{"edit":"allow","bash":"allow","webfetch":"deny"}}
EOF
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 \
      opencode run --format json -m "openrouter/$BENCH4_MODEL" "$(cat "$TASK")" \
      > "$OUT/harness.out" 2> "$OUT/harness.err"; rc=$? ;;
  mini)
    MSWEA_CONFIGURED=true MSWEA_COST_TRACKING=ignore_errors \
      /opt/mini/bin/mini -y --exit-immediately -l 0 -m "openrouter/$BENCH4_MODEL" -c mini.yaml -c agent.step_limit=1000 \
      -c "model.model_kwargs.api_base=$BENCH4_BASE/api/v1" -o "$OUT/trajectory.json" -t "$(cat "$TASK")" \
      > "$OUT/harness.out" 2> "$OUT/harness.err" < /dev/null; rc=$? ;;
  openhands)
    OPENHANDS_SUPPRESS_BANNER=1 BENCH4_BASE_URL="$BENCH4_BASE/api/v1" \
      /opt/oh/bin/python /opt/bench4/run_openhands.py "$TASK" 1000 \
      > "$OUT/harness.out" 2> "$OUT/harness.err"; rc=$? ;;
  *) echo "unknown harness $H" > "$OUT/harness.err"; rc=2 ;;
esac
echo "$rc" > "$OUT/rc"
exit 0
