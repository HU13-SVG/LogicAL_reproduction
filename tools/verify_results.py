#!/usr/bin/env python3
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / 'results' / 'localization_strict'
ORDER = ('breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors')
EXPECTED = {'breakfast_box': 78.03, 'juice_bottle': 93.7, 'pushpins': 78.26, 'screw_bag': 61.76, 'splicing_connectors': 63.41}
EXPECTED_MACRO = 75.03
EXPECTED_ROWS = 1136

def auc(labels, scores):
    ordered = sorted(zip(scores, labels))
    if any((not math.isfinite(float(s)) or int(y) not in (0, 1) for (s, y) in ordered)):
        raise ValueError('non-finite score or non-binary label')
    positives = sum((int(y) for (_, y) in ordered))
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('AUROC requires both positive and negative images')
    (rank_sum, start) = (0.0, 0)
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum((int(y) for (_, y) in ordered[start:end]))
        start = end
    return 100.0 * (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def main():
    results = json.loads((RESULT_DIR / 'results.json').read_text())
    if not results:
        raise SystemExit('results.json is empty')
    final = results[-1]
    if final.get('epoch') != 300:
        raise SystemExit(f"final evaluated epoch is {final.get('epoch')}, expected 300")
    rows = list(csv.DictReader(open(RESULT_DIR / 'scores_epoch_300.csv', newline='')))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f'expected {EXPECTED_ROWS} per-image rows, found {len(rows)}')
    if set(rows[0]) != {'category', 'label', 'score', 'path'}:
        raise SystemExit(f'unexpected CSV columns: {sorted(rows[0])}')
    labels = [int(r['label']) for r in rows]
    scores = [float(r['score']) for r in rows]
    cats = [r['category'] for r in rows]
    counts = Counter(cats)
    if set(counts) != set(ORDER):
        raise SystemExit(f'unexpected category set: {sorted(counts)}')
    for c in ORDER:
        n = counts[c]
        pos = sum((1 for (i, x) in enumerate(cats) if x == c and labels[i] == 1))
        neg = n - pos
        if pos == 0 or neg == 0:
            raise SystemExit(f'{c}: AUROC undefined (pos={pos}, neg={neg})')
    print(f"{'category':22s} {'n':>5s} {'stored':>8s} {'recomputed':>11s} {'expected':>9s}")
    ok = True
    per_category = {}
    for c in ORDER:
        idx = [i for (i, x) in enumerate(cats) if x == c]
        recomputed = auc([labels[i] for i in idx], [scores[i] for i in idx])
        per_category[c] = recomputed
        stored = float(final['per_category'][c]) * 100
        if abs(recomputed - stored) > 1e-09:
            ok = False
        same = abs(round(recomputed, 2) - EXPECTED[c]) < 1e-09
        ok = ok and same
        print(f"{c:22s} {len(idx):5d} {stored:8.2f} {recomputed:11.2f} {EXPECTED[c]:9.2f} {('OK' if same else 'MISMATCH')}")
    macro = sum(per_category.values()) / len(ORDER)
    stored_macro = float(final['logical_image_auroc_macro']) * 100
    if abs(macro - stored_macro) > 1e-09:
        ok = False
    same = abs(round(macro, 2) - EXPECTED_MACRO) < 1e-09
    ok = ok and same
    print(f"{'macro (mean of 5)':22s} {len(rows):5d} {stored_macro:8.2f} {macro:11.2f} {EXPECTED_MACRO:9.2f} {('OK' if same else 'MISMATCH')}")
    print()
    print('Result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
