"""Worker cards: per-(pool, model) size limits, standard library only.

Source is a short string naming where the limit was read (a provider doc URL, or
'measured: ledger/calls.jsonl').
"""
CARDS = {
    ("nvidia", "nemotron-3-ultra"): {
        "max_input_chars": 128000,
        "max_output_tokens": 16000,
        "source": "measured: ledger/calls.jsonl (Nemotron cut off at max_tokens=16000)",
    },
}


def check_size(pool: str, model: str, prompt: str, max_tokens: int):
    """Return ('ok', ''), ('split', reason), or ('refuse', reason)."""
    key = (pool, model)
    card = CARDS.get(key)
    if card is None:
        return "refuse", f"no card for {pool}/{model}"
    if max_tokens > card["max_output_tokens"]:
        return "refuse", f"max_tokens {max_tokens} > card max_output_tokens {card['max_output_tokens']}"
    if len(prompt) > card["max_input_chars"]:
        return "split", f"prompt {len(prompt)} chars > card max_input_chars {card['max_input_chars']}"
    return "ok", ""
