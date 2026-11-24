"""How many worker runs may go at once."""


def lanes(headroom, ready, cap=4):
    """Lanes from free budget headroom and ready queue items."""
    if ready == 0 or headroom < 0.2:
        return 0
    return min(cap, ready, int(headroom * cap) + 1)
