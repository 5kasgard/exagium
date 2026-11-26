"""Truth set Q1: items that depend on P21 (computed from queue.json)."""
from typing import List


def truth_q1(queue: dict) -> List[str]:
    """Return sorted IDs of items whose blocked_by field contains 'P21'."""
    result = []
    for item in queue.get("items", []):
        blocked_by = item.get("blocked_by", [])
        if "P21" in blocked_by:
            result.append(item["id"])
    return sorted(result)
