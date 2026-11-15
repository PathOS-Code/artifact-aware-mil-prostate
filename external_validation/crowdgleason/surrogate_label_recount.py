"""Read-only recount: does the CrowdGleason surrogate rule (prepare_crowdgleason_labels.py:
primary = most frequent cancer pattern, secondary = second most frequent, ties -> lower pattern)
give a different grade group than the ISUP needle-biopsy convention (secondary = highest other pattern
regardless of extent if higher grade; lower-grade pattern <5% not scored), using patch-count
proportions as a proxy for extent? Run from the repository root."""
import csv
from collections import Counter, defaultdict
G = {(3,3):1,(3,4):2,(4,3):3,(4,4):4,(3,5):4,(5,3):4,(4,5):5,(5,4):5,(5,5):5}
d = defaultdict(list)
for r in csv.DictReader(open('external_validation/labels/crowdgleason_patch_manifest.csv')):
    d[r['slide_id']].append(int(r['patch_gleason_pattern']))
two, changed = 0, []
for s, ps in d.items():
    c = Counter(p for p in ps if p > 0)
    if len(c) < 2:
        continue
    two += 1
    tot = sum(c.values())
    o = sorted(c.items(), key=lambda x: (-x[1], x[0]))
    sur = G[(o[0][0], o[1][0])]
    outs = set()
    for P in [k for k, v in c.items() if v == o[0][1]]:
        others = [k for k in c if k != P]
        hi = [k for k in others if k > P]
        lo = [k for k in others if c[k] / tot >= 0.05]
        Q = max(hi) if hi else (max(lo) if lo else P)
        outs.add(G[(P, Q)])
    if outs != {sur}:
        changed.append((s, dict(c), sur, sorted(outs)))
print('identifiers with >=2 cancer patterns:', two)
print('grade group differs from ISUP convention:', changed)
