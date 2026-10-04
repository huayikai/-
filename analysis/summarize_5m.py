"""Summarize paired BM/Router runs under a fixed 5M evaluation budget."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def window_mean(series, start, end):
    points = [(x, y) for x, y in series if math.isfinite(y)]
    if not points or points[-1][0] < end:
        return None
    xs, ys = map(np.asarray, zip(*points))
    inside = (xs > start) & (xs < end)
    x = np.concatenate(([start], xs[inside], [end]))
    y = np.concatenate(([np.interp(start, xs, ys)], ys[inside], [np.interp(end, xs, ys)]))
    return float(np.trapz(y, x) / (end - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'audit_update_2026-10-04/details.json.gz')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    details = json.loads(gzip.decompress(args.input.read_bytes()))
    rows = []
    for item in details:
        row = item['row']
        if not row['name'].startswith(('audit_router_both_', 'audit_bm_td_')):
            continue
        # Keep separately named ablations (e.g. both_nomix) out of the main pair.
        if row['name'] not in ('audit_router_both_{}_s{}'.format(row['map'], row['seed']),
                               'audit_bm_td_{}_s{}'.format(row['map'], row['seed'])):
            continue
        wins = item['series'].get('test_battle_won_mean', [])
        rows.append({'name': row['name'], 'map': row['map'], 'seed': row['seed'],
                     'auc_0_5m': window_mean(wins, 0, 5_000_000),
                     'win_4_5m': window_mean(wins, 4_000_000, 5_000_000),
                     'win_9_10m': row.get('win_9_10m')})
    output = args.output or args.input.parent / 'five_m_metrics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'runs': len(rows), 'full_5m': sum(r['win_4_5m'] is not None for r in rows),
                      'output': str(output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
