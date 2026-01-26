# CLAUDE.md — Development Guidelines

This document defines the **mandatory practices** for Python development in this repository. These are not suggestions — they are rules. Deviations must be justified with strong technical reasons and agreed upon in review.

---

## Python Package Management with `uv`

* **Use `uv` exclusively** for Python package management.
* Do **not** use `pip`, `pip-tools`, `poetry`, or `conda` directly.
* Commands you should know:

  * Install: `uv add <package>`
  * Remove: `uv remove <package>`
  * Sync lockfile: `uv sync`
* Running:

  * Python scripts: `uv run <script>.py`
  * Tools: `uv run pytest`, `uv run ruff`
  * REPL: `uv run python`

---

## Ad-hoc Analyses and One-Time Scripts

* **Use shell heredoc syntax** for one-time data analyses and exploratory work.
* Do **not** create throwaway `.py` files or use alternative shell tools (awk, sed, etc.) for data manipulation.
* Python is more readable, maintainable, and powerful for these tasks.

**Preferred pattern:**

```bash
uv run python3 << 'EOF'
import pandas as pd

# Your analysis code here
df = pd.read_csv('data.csv')
print(df.describe())
EOF
```

**Why this matters:**

* **No file clutter** — no orphaned `temp.py` or `test_script.py` files
* **Self-documenting** — the command and its context live together in shell history or docs
* **Efficient** — Claude can generate complete, working analyses inline
* **Reproducible** — easy to copy-paste entire commands

This approach is **mandatory** for:

* Quick data inspections
* One-time transformations
* Exploratory analyses
* Data quality checks

For **reusable** logic that runs regularly, create proper Python scripts or modules.

---

## Python Coding Style

### Type Hints (mandatory)

* Always annotate function parameters and return types.
* Use built-in generics (`list`, `dict`, `tuple`, `set`) — **never** import `List`, `Dict`, etc. from `typing`.
* Use `|` for unions (Python 3.10+).
* Annotate variables where type is not obvious.

```python
def process_data(items: list[str]) -> dict[str, int]:
    ...

value: str | None = None
```

### Error Handling

* **Never** use bare `except`.
* Always raise meaningful errors with context.
* Prefer explicit error classes over generic `Exception`.

### Logging

* Use `logging` — never `print` — for runtime diagnostics.

### Formatting & Linting

* Use `ruff` for both linting and formatting:

  * Format: `uv run ruff format`
  * Lint + fix: `uv run ruff check --fix`

---

## Testing Philosophy

Good tests do not just check code; they shape its design. Tests are **contracts**: they describe what must stay true even if the implementation changes.

### Golden Rules

1. **Test behavior, not implementation**

   * Assert on outputs and public APIs.
   Do not inspect private attributes like `_steps` unless no public API exists. If needed, add a public `.spec()` for testability.

2. **Each test should fail for one reason**

   * Keep assertions focused. Split broad tests into smaller ones.

3. **Prefer fast, deterministic tests**

   * No `sleep()`; control time with libraries like `freezegun` or dependency injection.
   * Control randomness by seeding or injecting RNGs.

4. **Use fakes over mocks**

   * Fake implementations are easier to read and maintain than heavy mocking.
   * Mock only at external boundaries (HTTP, file I/O, external services).

5. **Structure tests for readability**

   * Setup (Arrange) → Action → Assertion.
   * Use fixtures for repeated setup, but don’t hide complexity in `conftest.py`.

---

## Test Coverage

* Use `pytest-cov` to measure coverage.
* Run with coverage on every commit:

  * `uv run pytest --cov=src/transfer_learning_publication --cov-report=term-missing tests/`
* Coverage should be **used to find gaps**, not chased to 100%. A brittle 100% is worse than 85% meaningful coverage.

---

## Testing Conventions

### File & Class Organization

* **One test file per module**: `test_<module>.py`
* **One test class per function/class under test**: `Test<ThingUnderTest>`
* Test methods: descriptive, snake\_case, explain the behavior.
  Example: `test_fails_with_empty_dataframe`, not `test1`.

### Categories of Tests

1. **Basic functionality**: happy paths with simple inputs.
2. **Error handling**: invalid inputs should raise the right exception with the right message.
3. **Edge cases**: empty data, all-null columns, large inputs, weird types.
4. **Data preservation**: non-transformed columns, schema, and order remain intact.
5. **Integration paths**: small number of tests where the real pipeline runs end-to-end.

### Assert Patterns

* Prefer **direct comparisons** (`equals`, `==`) for DataFrames.
* For complex structures, use `.to_list()` or `.spec()` for clarity.
* Check types and schema explicitly when relevant.

```python
# Good
assert result["column"].to_list() == [1.0, 2.0, 3.0]
assert result.schema["col"] == pl.Float64
```

### Error Testing

Always assert both **exception type** and **message fragment**:

```python
with pytest.raises(ValueError, match="no steps"):
    builder.build()
```

### Fixtures

* Use fixtures sparingly and descriptively (`simple_df`, `df_with_missing_values`).
* Avoid fixture over-engineering; clarity > DRY.

---

## Advanced Testing Guidelines

* **Canonical Specs**: If an object has complex internal state, expose a `.spec()` or `.describe()` method that returns a plain dict/list summary. Test against that instead of private fields.
* **Lazy evaluation**: For Polars LazyFrame functions, assert that results remain lazy (don’t call `.collect()` unless necessary).

---

## Anti-Patterns (Avoid These)

Asserting on private attributes (`._steps`, `._fitted_steps`).
Overly specific error message checks (brittle wording).
Giant integration tests covering all cases — push most variation down into unit tests.
100s of trivial tests (getter/setter, boilerplate) — test behaviors that matter.
Hiding critical setup in nested fixtures.

In short: **tests should describe contracts, not internals.**
If your test breaks after a refactor that doesn’t change behavior, the test was wrong.

---

## Project Context

This repository contains experiments for **rainfall-runoff modeling in ungauged basins** using deep learning. The goal is to train models that can predict streamflow in locations without historical gauge data, based on meteorological inputs (precipitation, temperature, evaporation) and catchment attributes.

### Research Focus

* **Problem**: Prediction in Ungauged Basins (PUB) - estimating streamflow where no observations exist
* **Approach**: Deep learning models trained in simulation mode (concurrent prediction, not autoregressive)
* **Dataset**: CARAVAN global hydrological dataset with preprocessed time series
* **Test regions**: Madagascar/Mozambique and similar basins identified via PCA clustering
* **Models**: EA-LSTM and Mamba architectures (~1M parameters each)
* **Loss functions**: Comparing standard MSE, power loss, and variants with/without log transforms

### The `tl-` CLI Suite

This project uses the **transfer-learning-publication** package, which provides four CLIs for the complete ML workflow:

* **`tl-train`**: Train models from scratch with multi-seed support and automatic checkpointing
* **`tl-finetune`**: Fine-tune pre-trained models with reduced learning rates (transfer learning)
* **`tl-tune`**: Hyperparameter optimization using Optuna with auto-generated configs
* **`tl-evaluate`**: Evaluate trained models on test datasets with analysis-ready parquet outputs

**Documentation:**

* CLI guide: `/Users/nicolaslazaro/Desktop/work/transfer-learning-publication/docs/cli_guide.md`
* Configuration guide: `/Users/nicolaslazaro/Desktop/work/transfer-learning-publication/docs/configuration_guide.md`

### Repository Structure

```
pub-usa/
├── configs/
│   ├── models/              # Model configurations (data, features, architecture)
│   └── basin_ids_files/     # Text files with gauge IDs for basin selection
├── experiments/             # Experiment configurations (multi-model comparisons)
├── CLAUDE.md               # This file - development guidelines
└── README.md               # Project documentation
```

### Current Experiments

**Large Models (~1M parameters)**: Comparing EA-LSTM vs Mamba architectures

* Input window: 100 days of meteorological forcing
* Mode: Simulation (learns rainfall-runoff process without past streamflow)
* Basins: Madagascar/Mozambique cluster (identified via PCA on catchment attributes)
* Variants: Standard loss, power loss, with/without log transforms

### Configuration Files

All experiments are defined using YAML configuration files:

* **Model configs** (`configs/models/*.yaml`): Define data paths, features, sequences, model architecture, and training parameters
* **Experiment configs** (`experiments/*.yaml`): Define multi-model experiments with shared trainer settings
* **Basin lists** (`configs/basin_ids_files/*.txt`): One gauge ID per line for basin selection

**Key configuration sections:**

* `data`: Dataset location, basin selection, preprocessing pipeline
* `features`: Input features (forcing, static attributes), target variable
* `sequence`: Input/output window lengths
* `data_preparation`: Mode (simulation vs forecast), autoregressive settings
* `model`: Architecture type and hyperparameters
* `dataloader`: Batch size, number of workers, etc.
