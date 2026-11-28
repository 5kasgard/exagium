#!/usr/bin/env python3
"""Offline, deterministic inference-economics analyser (Exagium #258).

Consumes the two provider exports committed under
investigations/inference-economics-2026-11/ and prints a reconciled
summary. It performs NO network or paid inference calls: pure CSV arithmetic,
so it can run in CI or a free worker with no API budget.

Guardrails honoured (issue #258):
  * Anthropic and OpenRouter accounting stay SEPARATE; never joined by
    date/model, never summed as one ledger except an explicitly-labelled,
    opt-in balance reconciliation.
  * No secrets read or emitted.
  * Every figure reproducible from committed artefacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONSOLE = "claude_console_costs_by_model_token_type_2026-11-20_to_27.csv"
OPENROUTER = "openrouter_usage_nonzero.csv"

COST_COLS = {
    "input_no_cache_usd": "input_no_cache",
    "output_usd": "output",
    "input_cache_read_usd": "cache_read",
    "input_cache_write_5m_usd": "cache_write_5m",
    "input_cache_write_1h_usd": "cache_write_1h",
}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    return float(row.get(key) or 0.0)


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def analyse_console(rows):
    by_class, by_model, by_day = {}, {}, {}
    total = 0.0
    for row in rows:
        day_total = _f(row, "total_usd")
        total += day_total
        model = row["model"]
        by_model[model] = by_model.get(model, 0.0) + day_total
        by_day[row["usage_date_utc"]] = by_day.get(row["usage_date_utc"], 0.0) + day_total
        for col, cls in COST_COLS.items():
            by_class[cls] = by_class.get(cls, 0.0) + _f(row, col)

    cache_lifecycle = (by_class.get("cache_write_5m", 0.0)
                       + by_class.get("cache_write_1h", 0.0)
                       + by_class.get("cache_read", 0.0))
    write_total = by_class.get("cache_write_5m", 0.0) + by_class.get("cache_write_1h", 0.0)
    worst_day = max(by_day.items(), key=lambda kv: kv[1]) if by_day else ("", 0.0)
    return {
        "provider": "anthropic",
        "total_usd": round(total, 6),
        "by_token_class_usd": {k: round(v, 6) for k, v in sorted(by_class.items())},
        "cache_write_total_usd": round(write_total, 6),
        "cache_lifecycle_usd": round(cache_lifecycle, 6),
        "cache_lifecycle_pct": _pct(cache_lifecycle, total),
        "by_model_usd": {k: round(v, 6) for k, v in sorted(by_model.items())},
        "by_day_usd": {k: round(v, 6) for k, v in sorted(by_day.items())},
        "worst_day": worst_day[0],
        "worst_day_usd": round(worst_day[1], 6),
        "worst_day_pct": _pct(worst_day[1], total),
    }


def analyse_openrouter(rows):
    by_model, by_day = {}, {}
    total = 0.0
    for row in rows:
        usage = _f(row, "total_usage")
        total += usage
        by_model[row["model"]] = by_model.get(row["model"], 0.0) + usage
        by_day[row["created_at__day"]] = by_day.get(row["created_at__day"], 0.0) + usage
    top = max(by_model.items(), key=lambda kv: kv[1]) if by_model else ("", 0.0)
    return {
        "provider": "openrouter",
        "note": "total_usage treated as USD; ~balance delta (rounding only).",
        "total_usage_sum": round(total, 6),
        "by_model_usage": {k: round(v, 6) for k, v in sorted(by_model.items())},
        "by_day_usage": {k: round(v, 6) for k, v in sorted(by_day.items())},
        "top_model": top[0], "top_model_usage": round(top[1], 6),
        "top_model_pct": _pct(top[1], total),
    }


def reconcile(anthropic, openrouter, balances):
    if not balances:
        return None
    export_sum = anthropic["total_usd"] + openrouter["total_usage_sum"]
    observed = balances.get("observed_spend_usd")
    return {
        "warning": "Cross-provider sum shown for reconciliation only; not one ledger.",
        "export_sum_usd": round(export_sum, 6),
        "balance_observed_spend_usd": observed,
        "residual_usd": round(observed - export_sum, 6) if observed is not None else None,
    }


def print_human(console, openrouter, recon):
    print("\n== Anthropic Console (derived CSV) ==")
    print(f"total: ${console['total_usd']:.6f}")
    print("by token class (USD, % of total):")
    for cls, val in console["by_token_class_usd"].items():
        print(f"  {cls:16s} ${val:>10.6f}  {_pct(val, console['total_usd']):5.1f}%")
    print(f"cache writes total: ${console['cache_write_total_usd']:.6f} | "
          f"cache lifecycle (writes+reads): ${console['cache_lifecycle_usd']:.6f} "
          f"({console['cache_lifecycle_pct']}%)")
    print("by model (USD):")
    for model, val in console["by_model_usd"].items():
        print(f"  {model:20s} ${val:>10.6f}")
    print(f"worst day: {console['worst_day']} ${console['worst_day_usd']:.6f} ({console['worst_day_pct']}%)")
    print("\n== OpenRouter (non-zero rows) ==")
    print(f"total_usage sum: {openrouter['total_usage_sum']:.6f}")
    print(f"top model: {openrouter['top_model']} {openrouter['top_model_usage']:.6f} ({openrouter['top_model_pct']}%)")
    for model, val in openrouter["by_model_usage"].items():
        print(f"  {model:26s} {val:>10.6f}")
    if recon:
        print("\n== Cross-provider reconciliation (NOT one ledger) ==")
        print(json.dumps(recon, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Offline inference-economics analyser (#258)")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--balances", help='optional JSON e.g. {"observed_spend_usd": 18.64}')
    args = ap.parse_args()
    base = Path(args.dir)
    console = analyse_console(_rows(base / CONSOLE))
    openrouter = analyse_openrouter(_rows(base / OPENROUTER))
    balances = json.loads(args.balances) if args.balances else None
    recon = reconcile(console, openrouter, balances)
    if args.json:
        print(json.dumps({"anthropic": console, "openrouter": openrouter, "reconcile": recon}, indent=2))
    else:
        print_human(console, openrouter, recon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
