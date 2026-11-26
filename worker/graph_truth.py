"""Truth set Q1: ids of queue items whose dependency field names P21."""
TARGET = "P21"
DEP_FIELDS = ("blocked_by", "depends_on", "dependencies", "dependency")


def _depends_on_target(item):
    for field in DEP_FIELDS:
        value = item.get(field)
        if isinstance(value, str):
            if value.strip() == TARGET:
                return True
        elif isinstance(value, (list, tuple, set)):
            if TARGET in value:
                return True
    return False


def truth_q1(queue):
    """Sorted ids of items that depend on P21 (from the parsed queue)."""
    return sorted(item["id"] for item in queue.get("items", []) if _depends_on_target(item))
