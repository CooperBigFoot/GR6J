# Claude Agent Guidelines

## Project Overview

Python implementation of the GR6J lumped conceptual rainfall-runoff model for daily streamflow simulation. Extends GR4J with an exponential store for improved low-flow modeling, and includes the optional CemaNeige snow module for cold-climate catchments. Supports both simulation and automatic calibration via evolutionary algorithms.

## Quick Reference

```bash
# Package management
uv add <package>          # Install
uv remove <package>       # Remove
uv sync                   # Sync lockfile

# Running code
uv run python <script>    # Scripts
uv run pytest             # Tests
uv run ruff format        # Format
uv run ruff check --fix   # Lint

# Test coverage
uv run pytest --cov=src/<package> --cov-report=term-missing tests/
```

## Core Principles

### Task Focus

- **No task jags**: Don't switch from implementing A → testing A → implementing B mid-stream
- **Delegate aggressively**: 3+ sub-agents at a time for orthogonal tasks
- **Stay high-level**: Coordinate, don't jump in for quick fixes

### Context Gathering

- `.context/` contains library submodules—grep for implementation details before coding
- Ask clarifying questions upfront rather than implementing wrong solutions

---

## Python Standards

### Type Hints (mandatory)

```python
def process(items: list[str], limit: int | None = None) -> dict[str, int]:
    ...
```

- Use built-in generics (`list`, `dict`)—never import from `typing`
- Use `|` for unions (not `Optional`)

### Code Style

- `logging` over `print`
- No bare `except:`—always specify exception type
- Shallow nesting (max 2-3 levels)
- Prefer comprehensions and pipelines over nested loops

### Documentation

**Code clarity + type hints = primary documentation.** Docstrings only when:

- Code is not self-explanatory
- User-facing API (public functions/methods)

Use Google style, no `Examples` section:

```python
def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency coefficient.

    Args:
        observed: Array of observed values.
        simulated: Array of simulated values, same length as observed.

    Returns:
        NSE value in range (-∞, 1], where 1 is perfect fit.

    Raises:
        ValueError: If arrays have different lengths.
    """
```

### Ad-hoc Analysis (no temp files)

```bash
uv run python3 << 'EOF'
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
EOF
```

---

## Testability

### Inject Dependencies (critical)

```python
# ❌ Untestable
def create_record():
    return {"created_at": datetime.now()}

# ✅ Testable
def create_record(clock: Callable[[], datetime]) -> dict:
    return {"created_at": clock()}
```

### Testing Rules

1. Test behavior, not implementation—no asserting on private attributes
2. Use fakes over mocks (mock only at external boundaries)
3. Assert exception type AND message: `pytest.raises(ValueError, match="no steps")`
4. One test file per module: `test_<module>.py`
5. Descriptive names: `test_fails_with_empty_dataframe`

---

## Feature Implementation Workflow

**Phases must be completed in order. No skipping.**

| Phase | Action | Gate |
|-------|--------|------|
| 1. Requirements | Ask clarifying questions | Full scope understood |
| 2. Contracts | Design `Protocol`, `dataclass`, type signatures | User approval |
| 3. Test Scaffold | Write test structure (bodies = `pass`) | Coverage verified |
| 4. Test Bodies | Implement tests using contracts only | Tests compile |
| 5. Implementation | Parallel agents implement logic | All tests pass |

**Rules:**

- No implementation code until Phase 5
- Contract changes after Phase 2 require user approval with justification
- Phases 4-5: spawn 10+ parallel agents

---

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Assert on `._private` attributes | Use public API or add `.spec()` |
| Bare `except:` | Specific exception types |
| `from typing import List, Dict` | Built-in `list`, `dict` |
| `print()` for diagnostics | `logging` |
| Temp `.py` files for one-offs | Heredoc syntax |
| Docstrings on every function | Only non-obvious or public API |
| `Examples` section in docstrings | Clear Args/Returns/Raises only |
| `datetime.now()` in business logic | Inject clock dependency |
