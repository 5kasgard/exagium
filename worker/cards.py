"""Worker cards: pure size check per (pool, model). Stdlib only."""
from __future__ import annotations

CARDS: dict[tuple[str, str], dict[str, int | str]] = {
    # (pool, model): {max_input_chars, max_output_tokens, source}
    # Only limits with a citable source are included.
    ("nvidia", "nvidia/nemotron-3-ultra"): {
        "max_input_chars": 128000,
        "max_output_tokens": 16000,
        "source": "observed 2026-11-19: finish_reason=length at max_tokens=16000",
    },
    # Add more cards here as sources become available.
}

def check_size(pool: str, model: str, prompt_chars: int, max_tokens: int):
    """Return ('ok', None), ('split', parts), or ('refuse', reason)."""
    card = CARDS.get((pool, model))
    if card is None:
        return ("refuse", f"no size card for ({pool}, {model})")
    max_in = card.get("max_input_chars")
    max_out = card.get("max_output_tokens")
    if max_out is not None and max_tokens > max_out:
        return ("refuse", f"max_tokens {max_tokens} > output limit {max_out}")
    if max_in is not None and prompt_chars > max_in:
        parts = (prompt_chars + max_in - 1) // max_in
        return ("split", parts)
    return ("ok", None)
