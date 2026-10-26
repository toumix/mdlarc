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
| `high-1` | 2026-10-26 | `run_script.py high` unchanged, one Modal H100, torch 2.10, flash-attn 2.8.3, seed 42 | ~44% (README; the author's 5090 and software differ, so numerics do) | — |
