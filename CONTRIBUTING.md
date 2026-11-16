# Contributing

Use Python 3.9–3.12, create a virtual environment, and install the development
dependencies with `python -m pip install -e ".[dev]"`.

Before opening a pull request, run:

```bash
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m build
```

Describe the problem, the resulting behavior, and how you checked it. For a new
algorithm, add its implementation under `src/best_arm/algorithms`, register it in
`model.py`, document constructor options and compatible environments, and add a
small seeded smoke case. Use `run()`'s existing result tuple; if you implement
`run_fixed()`, document the bootstrap and recorded-history shapes.

Numerical changes should explain which equations or algorithm steps change and
include a regression case. Preserve random-draw order when making formatting or
documentation-only changes. A smoke test demonstrates execution and result
invariants; confidence guarantees need separate mathematical and empirical work.

Keep generated plots, experiment results, caches, environments, editor settings,
credentials, and unfinished research drafts out of commits. Use small synthetic
JSON fixtures in `examples/environments` for shareable demonstrations.
