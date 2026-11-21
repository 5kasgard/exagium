"""Parse ROUTER_USAGE_LOG into per-attempt cost records. Standard library only."""
import json


def parse_calls(lines):
    by_attempt = {}
    skipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        usage = rec.get("usage", {})
        cost = usage.get("cost")
        if cost is None:
            skipped += 1
            continue
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        attempt_id = rec.get("attempt_id")
        if attempt_id not in by_attempt:
            by_attempt[attempt_id] = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        by_attempt[attempt_id]["cost"] += cost
        by_attempt[attempt_id]["input_tokens"] += input_tokens
        by_attempt[attempt_id]["output_tokens"] += output_tokens
        by_attempt[attempt_id]["calls"] += 1
    return by_attempt, skipped
