#!/usr/bin/env python3
"""Generate views of the queue grouped by touched files/directories.
If QUEUE.md with the overlap table exists, it is parsed; otherwise the
queue.json is used (requires the same format as the live queue).
Outputs JSON and Markdown reports in the same directory.
"""
import json, re, sys
from pathlib import Path

def parse_queue_md(md_path: Path):
    text = md_path.read_text()
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith('| item'):
            in_table = True
            continue
        if in_table:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 5:
                rows.append(parts)
            else:
                break
    path_map = {}
    batch_map = {}
    item_map = {}
    for item, batch, item2, batch2, both in rows:
        touches = [t.strip() for t in both.split(',')]
        for t in touches:
            path_map.setdefault(t, set()).update([item, item2])
        batch_map.setdefault(batch, set()).add(item)
        batch_map.setdefault(batch2, set()).add(item2)
        item_map.setdefault(item, set()).update(touches)
        item_map.setdefault(item2, set()).update(touches)
    # Convert sets to sorted lists
    path_map = {k: sorted(v) for k, v in path_map.items()}
    batch_map = {k: sorted(v) for k, v in batch_map.items()}
    item_map = {k: sorted(v) for k, v in item_map.items()}
    return path_map, batch_map, item_map

def main():
    base = Path('audit/2026-11-17')
    md = base / 'QUEUE.md'
    if not md.exists():
        print('QUEUE.md not found', file=sys.stderr)
        sys.exit(1)
    path_map, batch_map, item_map = parse_queue_md(md)
    out_json = base / 'graph_by_path.json'
    out_md = base / 'graph_by_path.md'
    out_json.write_text(json.dumps({'by_path': path_map, 'by_batch': batch_map, 'by_item': item_map}, indent=2))
    lines = ['# Queue grouping by touched paths', '']
    for path, items in path_map.items():
        lines.append(f"## {path}\n")
        lines.append('Items: ' + ', '.join(items) + '\n')
    out_md.write_text('\n'.join(lines))
    print('Generated', out_json, 'and', out_md)

if __name__ == '__main__':
    main()
