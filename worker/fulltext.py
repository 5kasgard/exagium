"""Full-text helper: arXiv HTML URL and results-section extraction."""
import re

from research_job import strip_html   # same rules as the research job


def arxiv_html_url(arxiv_id):
    return "https://arxiv.org/html/" + str(arxiv_id)


HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.I | re.S)


def results_text(html, max_chars):
    """Text from headings containing 'result' or 'experiment', up to the next heading of the same or higher level."""
    h = html or ""
    heads = [(m.start(), m.end(), int(m.group(1)), strip_html(m.group(2)))
             for m in HEADING.finditer(h)]
    spans = []
    for i, (start, end, level, text) in enumerate(heads):
        if re.search(r"result|experiment", text, re.I):
            stop = len(h)
            for j in range(i + 1, len(heads)):
                if heads[j][2] <= level:
                    stop = heads[j][0]
                    break
            spans.append((end, stop))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append([a, b])
    parts = [strip_html(h[a:b]) for a, b in merged]
    return " ".join(p for p in parts if p)[:max_chars]
