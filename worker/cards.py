"""Worker cards: per-(pool, model) size limits with cited sources. Stdlib only."""
import math

# CARDS[(pool, model)] = {
#     "max_input_chars": int,
#     "max_output_tokens": int,
#     "source": "provider doc URL" | "observed YYYY-MM-DD: evidence",
# }
# Only enter a limit if it can be cited. Missing key -> no card -> refuse.
CARDS = {
    ("nvidia", "nvidia/nemotron-3-ultra"): {
        "max_input_chars": 128_000,
        "max_output_tokens": 16_000,
        "source": "observed 2026-11-19: finish_reason='length' at max_tokens=16000 on nvidia/nemotron-3-ultra",
    },
}


def check_size(pool: str, model: str, prompt_chars: int, max_tokens: int):
    """Return ('ok', None), ('split', parts), or ('refuse', reason)."""
    key = (pool, model)
    card = CARDS.get(key)
    if card is None:
        return ("refuse", f"no card for {pool}/{model}")
    if max_tokens > card["max_output_tokens"]:
        return ("refuse", f"max_tokens {max_tokens} > output limit {card['max_output_tokens']}")
    limit = card["max_input_chars"]
    if prompt_chars <= limit:
        return ("ok", None)
    parts = math.ceil(prompt_chars / limit)
    return ("split", parts)
