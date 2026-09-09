## What does this change?

<!-- One or two sentences on what and why. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -q -m "not weights"` passes
- [ ] Tests added/updated for behavior changes
- [ ] If this touches `adapters/`, `calibrate.py`, or `truncate.py`: I've read
      the invariants in [CONTRIBUTING.md](../CONTRIBUTING.md#invariants-that-are-easy-to-break-silently)
      and the change respects them (or the change is to one of them, explained above)
