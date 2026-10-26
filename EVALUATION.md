# Evaluation log

The ARC-AGI-1 evaluation test outputs (`assets/solutions.json`) are sealed: a run reads them only
to score a claim logged here first, once, and the number is reported whatever it is. Training uses
the evaluation tasks' demonstration pairs, never their test outputs, as TRM does. The smoke runs
set `MDLARC_SCORE=0` and never open the file.

The build is the README's `build_datasets.py arc1 --add-conceptarc --with-filtered`, downloaded
fresh in the image: `challenges.json` sha256 `d9bf3195…`, `solutions.json` `629622b0…`.
`audit.py` compares its 907 training tasks with the 400 evaluation tasks grid by grid: one test
pair is shared (`070dd51e`, a duplicate of ARC-AGI-1's own training task `40853293`), and one
3×3 single-colour test output also occurs in ConceptARC (`3194b014`).

| claim | logged | what runs | expected | read |
|---|---|---|---|---|
| `high-1` | 2026-10-26 | `run_script.py high` unchanged, one Modal H100, torch 2.10, flash-attn 2.8.3, seed 42 | ~44% (README; the author's 5090 and software differ, so numerics do) | **162.5/400 = 40.6%**, 161 tasks fully solved; 4,392 s on the H100 (~1 h training, 688 s inference) |

`high-1` against wave 1 (rel-int/arc_1d#3, whose per-task sets were already read): of the 28 tasks
its symbolic methods solve (library of re-arc verifiers 21, enumeration 7, the LDT's 3 new
programs), 17 are among mdlARC's 161 and **11 are not**, two of the LDT's three among them
(`0c786b71`, `d4b1c2b1`). The union is 172 tasks, an oracle bound until a rule that picks between
them is logged and read.
