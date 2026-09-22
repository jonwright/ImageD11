# Agent notes

## Ground rules — never fudge the physics

- Never change the **physics / crystallography** in order to please, make a test pass, or match an expectation.
- Never **invent figures, numbers, or merit** that are not derived from the data or the actual computation.
- Never **invent "workarounds"** (e.g. adding tolerated slack, scaling metrics, or special-casing inputs) to make a result look better. If the geometry or matching is wrong, fix the real cause.

If a result "looks bad", investigate the actual cause (geometry, calibration, indexing, data quality) rather than relaxing criteria or adding ad-hoc corrections.

## Python 2.7 compatibility

The codebase must stay importable on **Python 2.7**. When editing, do **not** use py3-only syntax:

- No `@` **matrix-multiplication** operator (e.g. `a @ b`) — use `numpy.dot(a, b)` instead. (`@` decorators on functions/classes are fine; the infix operator is not.)
- No **f-strings** (`f"..."`) — use `"{}".format(...)` or `%` formatting.
- No bare `print(...)` — either avoid printing, or add `from __future__ import print_function` to the file.
- No other py3-only features (e.g. `yield from`, keyword-only arguments, `nonlocal`, type annotations on 2.7-builtins) without guarding by version.

Run tests under the py3 environment, but keep the syntax 2.7-compatible.

