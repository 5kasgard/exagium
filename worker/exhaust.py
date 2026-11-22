"""Parse a 429 reply into a pool reset time."""
import re

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def reset_time(status, headers, now):
    """Return None unless status is 429; otherwise now + retry-after seconds, defaulting to 60."""
    if status != 429:
        return None
    value = (headers or {}).get("retry-after")
    match = _NUM.search(str(value)) if value is not None else None
    if not match:
        return now + 60
    try:
        seconds = float(match.group())
    except ValueError:
        return now + 60
    if seconds < 0:
        return now + 60
    return now + seconds
