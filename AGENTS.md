# Agent notes

## Ground rules — never fudge the physics

- Never change the **physics / crystallography** in order to please, make a test pass, or match an expectation.
- Never **invent figures, numbers, or merit** that are not derived from the data or the actual computation.
- Never **invent "workarounds"** (e.g. adding tolerated slack, scaling metrics, or special-casing inputs) to make a result look better. If the geometry or matching is wrong, fix the real cause.

If a result "looks bad", investigate the actual cause (geometry, calibration, indexing, data quality) rather than relaxing criteria or adding ad-hoc corrections.
