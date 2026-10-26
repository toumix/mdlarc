"""Content-level leakage audit of mdlARC's `arc1 --add-conceptarc --with-filtered` build:
does any ARC-AGI-1 evaluation test output (or test pair) appear in the training tasks?
`build_datasets.py` filters ARC-AGI-2 by task id only; this compares the grids.

    python -I audit.py <ARC-AGI/data> <ARC-AGI-2/data> [<ConceptARC/corpus>]
"""
import json, sys
from pathlib import Path
A1, A2 = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda d: {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
a1t, a1e = load(A1/"training"), load(A1/"evaluation")
a2t, a2e = load(A2/"training"), load(A2/"evaluation")
a1, a2 = {**a1t, **a1e}, {**a2t, **a2e}
dup = set(a1) & set(a2)
print("id overlap ARC-1/ARC-2:", len(dup), "| eval-1 ids in ARC-2:", len(set(a1e) & set(a2)))
C = Path(sys.argv[3]) if len(sys.argv) > 3 else None
concept = {p.stem: json.loads(p.read_text()) for p in sorted(C.rglob("*.json"))} if C else {}
train = {**{k: v for k, v in a1t.items()}, **{f"c:{k}": v for k, v in concept.items()},
         **{f"a2:{k}": v for k, v in a2.items() if k not in dup}}
g = lambda x: json.dumps(x, separators=(",", ":"))
grids, pairs, inputs = {}, {}, {}
for tid, t in train.items():
    for ex in t["train"] + t["test"]:
        grids.setdefault(g(ex["output"]), set()).add(tid)
        inputs.setdefault(g(ex["input"]), set()).add(tid)
        pairs.setdefault(g([ex["input"], ex["output"]]), set()).add(tid)
hit_pair, hit_out, hit_demo = {}, {}, {}
for tid, t in a1e.items():
    for ex in t["test"]:
        if g([ex["input"], ex["output"]]) in pairs: hit_pair.setdefault(tid, set()).update(pairs[g([ex["input"], ex["output"]])])
        o = ex["output"]
        if len(o) * len(o[0]) >= 9 and g(o) in grids: hit_out.setdefault(tid, set()).update(grids[g(o)])
    shared = [g(ex["input"]) in inputs for ex in t["train"]]
    if sum(shared) >= 2: hit_demo[tid] = sum(shared)
print("training tasks:", len(train), "(ARC-1 train", len(a1t), "ConceptARC", len(concept), "ARC-2 filtered", len(train)-len(a1t)-len(concept), ")")
print("ARC-1 eval tasks whose TEST PAIR is in training:", len(hit_pair))
for k, v in sorted(hit_pair.items()): print("  ", k, sorted(v)[:4])
print("ARC-1 eval tasks whose TEST OUTPUT (>=3x3) is in training:", len(hit_out))
for k, v in sorted(hit_out.items()): print("  ", k, sorted(v)[:4])
print("ARC-1 eval tasks with >=2 demo inputs found in training:", len(hit_demo), sorted(hit_demo))
