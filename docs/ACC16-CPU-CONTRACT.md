# ACC16 shared budget method token — 2026-10-13

Initial observed state: clean `wip/release-readiness-2026-10-11`; no
repository-local AGENTS.md was found; research workspace instructions apply.

Objective: allow the shared immutable budget/runtime to describe a Taylor
adapter without claiming that this package implements the Taylor formula. The
TRELLIS.2 consumer owns its forecast implementation and CPU contract.

Change: added `taylor` to the method-neutral backend/fallback token set and
kept validation messages derived from that set. No forecasting formula or
adapter behavior was added here.

Verification (Windows, Python 3.12.10): `python tests/run_tests.py` completed
with **ALL MODULES PASSED** while the temporary budget-runtime contract cases
were registered; existing checkpoint, DIT forecast and H1 adapter checks also
passed. The temporary unit-test file and runner registration were then removed
in accordance with the owner's repository-wide instruction not to retain unit
tests. `compileall` and `git diff --check` passed. The consumer-side Taylor
contract suite is recorded in the fast-trellis2 checkout's
`docs/ACC16-CPU-CONTRACT.md`; its temporary unit-test files were also removed.

Evidence limit: shared API compatibility is not evidence of GPU/model
compatibility, generated-asset quality, memory peak, or speedup. Taylor remains
an operational method label only; consumers must implement and validate their
own forecast and fallback semantics.
