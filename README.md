# PyDrology

Lumped conceptual rainfall-runoff modeling at daily timesteps, powered by a compiled Rust core.

## Maintenance Status

🟢 **Active Development**
This repository is part of an ongoing project and actively maintained.

**Topic:** `python-package`

## Architecture

```
Python API  →  PyO3 bindings  →  Rust core
(pydrology)    (pydrology._core)  (pydrology-core)
```

All numerical model code (GR2M, GR6J, HBV-Light, CemaNeige) runs in compiled Rust. The Python layer provides parameter/state dataclasses, forcing data handling, calibration, and result formatting. PyO3/maturin bridges the two at the numpy array level.

## Installation

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and a [Rust toolchain](https://rustup.rs/) (for building the native extension).

```bash
git clone https://github.com/CooperBigFoot/pydrology.git
cd pydrology
uv sync            # install Python deps + build Rust extension via maturin
```

The first build compiles the Rust core and may take 1-2 minutes. Subsequent builds are incremental and much faster.

## Quick Start

```python
import numpy as np
from pydrology import ForcingData, list_models, get_model

# See available models
print(list_models())  # ['gr2m', 'gr6j', 'gr6j_cemaneige', 'hbv_light']

# Load a model
gr6j = get_model("gr6j")

# Prepare forcing data
forcing = ForcingData(
    time=np.arange("2000-01-01", "2005-01-01", dtype="datetime64[D]"),
    precip=np.random.uniform(0, 20, 1827),
    pet=np.random.uniform(1, 5, 1827),
)

# Run simulation
params = gr6j.Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
output = gr6j.run(params, forcing)

# Access results
print(output.streamflow)       # numpy array
print(output.to_dataframe())   # pandas DataFrame
```

## Calibration

```python
from pydrology import calibrate, ObservedData, Resolution, list_metrics

# Available metrics: nse, kge, log_nse, rmse, mae, pbias
print(list_metrics())

observed = ObservedData(
    time=forcing.time[365:],      # After warmup
    streamflow=observed_array,
)

# Single-objective
result = calibrate(
    model="gr6j",
    forcing=forcing,
    observed=observed,
    objectives=["nse"],
    use_default_bounds=True,
)
print(result.parameters, result.score)

# Multi-objective returns Pareto front
pareto = calibrate(..., objectives=["nse", "log_nse"])

# Cross-resolution: monthly observed vs daily forcing
monthly_obs = ObservedData(
    time=monthly_dates,
    streamflow=monthly_flows,
    resolution=Resolution.monthly,
)
result = calibrate(
    model="gr6j",
    forcing=daily_forcing,
    observed=monthly_obs,
    objectives=["nse"],
    use_default_bounds=True,
    observed_aggregation="sum",  # or "mean"
)
```

## Resolution Support

ForcingData supports multiple temporal resolutions with built-in validation and aggregation:

```python
from pydrology import Resolution, ForcingData

# Create daily forcing (default)
forcing = ForcingData(
    time=time_array,
    precip=precip_array,
    pet=pet_array,
    resolution=Resolution.daily,  # Default, can be omitted
)

# Aggregate to monthly (requires polars: uv add polars)
monthly_forcing = forcing.aggregate(Resolution.monthly)
```

Available resolutions: `hourly`, `daily`, `monthly`, `annual`. Currently all models support daily resolution.

## Snow Models

For snow-influenced catchments, use `gr6j_cemaneige` or `hbv_light`:

```python
from pydrology import Catchment

model = get_model("gr6j_cemaneige")
params = model.Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5, ctg=0.97, kf=2.5)
catchment = Catchment(mean_annual_solid_precip=150.0)

# Forcing must include temperature
forcing = ForcingData(time=..., precip=..., pet=..., temp=...)
output = model.run(params, forcing, catchment)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Three-layer design, trait system, adding new models
- [User Guide](docs/USER_GUIDE.md) — Custom states, multi-layer elevation bands, advanced usage
- [GR2M Model](docs/GR2M.md) — Monthly model with 2 parameters
- [GR6J Model](docs/GR6J.md) — Parameter details and equations
- [HBV-light Model](docs/HBV_LIGHT.md) — Parameter details and structure

## Development

```bash
uv run maturin develop --release   # rebuild Rust extension (release mode)
cargo test --workspace             # run Rust unit tests
uv run python -m pytest            # run Python test suite
uv run ruff format && uv run ruff check --fix  # format + lint
```

## References

- Pushpalatha et al. (2011). [GR6J model](https://doi.org/10.1016/j.jhydrol.2011.07.028)
- Valéry et al. (2014). [CemaNeige snow module](https://doi.org/10.1016/j.jhydrol.2014.03.064)
