# Adding a New Hydrological Model to PyDrology

Self-contained tutorial for adding a new hydrological model. Every section has
copy-paste-ready code with `<PLACEHOLDER>` markers. Primary audience: LLM agents.

---

## 1. Decision Tree

**Q1: Does the model fit the `HydrologicalModel` trait?**
- The trait works for single-zone models with fixed params + fixed forcing (precip + pet).
- YES: implement `HydrologicalModel` with associated types (like GR2M, GR6J).
- NO: write standalone `run()` / `step()`, skip the trait (like GR6J-CemaNeige).

**Q2: Does it need precomputed context?**
- Context = data derived from params, constant for a given run (e.g., UH ordinates).
- YES: define `<Model>Context` struct, implement `prepare()` (like GR6J).
- NO: use `type Context = ()` and `fn prepare() {}` (like GR2M).

**Q3: Is state fixed-size?**
- YES: use fixed-size arrays + `#[derive(Copy)]` (like GR2M: 2, GR6J: 63).
- NO: use `compute_state_size()` pattern (like HBV-Light with variable zones).

| Model | Fits Trait? | Has Context? | Fixed State? |
|-------|-------------|--------------|--------------|
| GR2M | Yes | No (`()`) | Yes (2) |
| GR6J | Yes | Yes (UH ordinates) | Yes (63) |
| HBV-Light | Yes (single-zone) | Yes (UH weights) | No (varies by zones) |
| GR6J-CemaNeige | No (extra args) | N/A | No (varies by layers) |

---

## 2. File Inventory

Replace `<model>` with snake_case (e.g., `gr4h`), `<Model>` with PascalCase.

**Files to CREATE:**
```
crates/pydrology-core/src/<model>/mod.rs
crates/pydrology-core/src/<model>/constants.rs
crates/pydrology-core/src/<model>/params.rs
crates/pydrology-core/src/<model>/state.rs
crates/pydrology-core/src/<model>/fluxes.rs
crates/pydrology-core/src/<model>/processes.rs
crates/pydrology-core/src/<model>/run.rs
crates/pydrology-python/src/<model>.rs
python/pydrology/models/<model>.py
tests/models/<model>/__init__.py
tests/models/<model>/test_run.py
tests/models/<model>/test_types.py
```

**Files to EDIT:**
```
crates/pydrology-core/src/lib.rs          -- add `pub mod <model>;`
crates/pydrology-python/src/lib.rs        -- 3 edits (mod, register, sys.modules)
python/pydrology/__init__.py              -- add import trigger
tests/test_model_conformance.py           -- add to ALL_MODELS + fixtures
```

---

## 3. Rust Core Templates

### 3.1 mod.rs -- `crates/pydrology-core/src/<model>/mod.rs`

```rust
/// <MODEL_NAME> -- <MODEL_DESCRIPTION>.
pub mod constants;
pub mod fluxes;
pub mod params;
pub mod processes;
pub mod run;
pub mod state;
```

### 3.2 constants.rs -- `crates/pydrology-core/src/<model>/constants.rs`

```rust
/// <MODEL_NAME> numerical constants and model contract.
use crate::forcing::Resolution;

// -- Numerical safeguards --
pub const <CONSTANT_NAME>: f64 = <VALUE>;

// -- Model contract constants --
pub const PARAM_NAMES: &[&str] = &[<"param1", "param2">];
pub const STATE_SIZE: usize = <N>;
pub const SUPPORTED_RESOLUTIONS: &[Resolution] = &[Resolution::Daily];

// -- Parameter bounds --
pub struct Bounds { pub min: f64, pub max: f64 }

pub const <PARAM1>_BOUNDS: Bounds = Bounds { min: <MIN>, max: <MAX> };
pub const <PARAM2>_BOUNDS: Bounds = Bounds { min: <MIN>, max: <MAX> };

pub const PARAM_BOUNDS: &[(f64, f64)] = &[
    (<MIN>, <MAX>), // <param1>
    (<MIN>, <MAX>), // <param2>
];
```

RULES: `PARAM_BOUNDS` tuple order MUST match `PARAM_NAMES` order. `Bounds` struct MUST be defined. `SUPPORTED_RESOLUTIONS` MUST be exactly one of `Daily` or `Monthly`.

### 3.3 params.rs -- `crates/pydrology-core/src/<model>/params.rs`

```rust
/// <MODEL_NAME> calibrated parameters.
use super::constants::{PARAM_BOUNDS, PARAM_NAMES};
use crate::traits::ModelParams;

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub <param1>: f64,
    pub <param2>: f64,
}

impl Parameters {
    pub fn new(<param1>: f64, <param2>: f64) -> Self {
        Self { <param1>, <param2> }
    }
}

impl ModelParams for Parameters {
    const N_PARAMS: usize = <N>;
    const PARAM_NAMES: &'static [&'static str] = PARAM_NAMES;
    const PARAM_BOUNDS: &'static [(f64, f64)] = PARAM_BOUNDS;

    fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != Self::N_PARAMS {
            return Err(format!("expected {} parameters, got {}", Self::N_PARAMS, arr.len()));
        }
        Ok(Self::new(arr[0], arr[1]))
    }

    fn to_array(&self) -> Vec<f64> {
        vec![self.<param1>, self.<param2>]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelParams;

    #[test]
    fn from_array_valid() {
        let p = Parameters::from_array(&[<val1>, <val2>]).unwrap();
        assert_eq!(p.<param1>, <val1>);
    }

    #[test]
    fn from_array_wrong_length() {
        assert!(Parameters::from_array(&[<val1>]).is_err());
    }

    #[test]
    fn to_array_roundtrip() {
        let p = Parameters::new(<val1>, <val2>);
        let p2 = Parameters::from_array(&p.to_array()).unwrap();
        assert_eq!(p.<param1>, p2.<param1>);
    }
}
```

RULES: All fields MUST be `pub f64`. `new()`, `from_array`, `to_array` order MUST match `PARAM_NAMES`. ALWAYS derive `Debug, Clone, Copy`.

### 3.4 state.rs -- `crates/pydrology-core/src/<model>/state.rs`

```rust
/// <MODEL_NAME> model state variables.
use super::constants::STATE_SIZE;
use super::params::Parameters;
use crate::traits::ModelState;

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub <store1>: f64,
    pub <store2>: f64,
}

impl State {
    pub fn initialize(params: &Parameters) -> Self {
        Self {
            <store1>: 0.3 * params.<capacity_param>,
            <store2>: 0.3 * params.<capacity_param>,
        }
    }
}

impl ModelState for State {
    fn to_vec(&self) -> Vec<f64> {
        vec![self.<store1>, self.<store2>]
    }

    fn from_slice(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != STATE_SIZE {
            return Err(format!("expected {} state elements, got {}", STATE_SIZE, arr.len()));
        }
        Ok(Self { <store1>: arr[0], <store2>: arr[1] })
    }

    fn array_len(&self) -> usize { STATE_SIZE }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelState;

    #[test]
    fn to_vec_from_slice_roundtrip() {
        let s = State::initialize(&Parameters::new(<val1>, <val2>));
        let s2 = State::from_slice(&s.to_vec()).unwrap();
        assert_eq!(s.<store1>, s2.<store1>);
    }

    #[test]
    fn from_slice_wrong_length() {
        assert!(State::from_slice(&[1.0]).is_err());
    }
}
```

RULES: Use `Copy` only for small fixed-size state. `to_vec()` and `from_slice()` order MUST be identical.

### 3.5 fluxes.rs -- `crates/pydrology-core/src/<model>/fluxes.rs`

```rust
/// <MODEL_NAME> model flux outputs.
///
/// `FluxesTimeseries` is auto-generated by the `Fluxes` derive macro.
use pydrology_macros::Fluxes;
use crate::traits::FluxesTimeseriesOps;

#[derive(Debug, Clone, Copy, Fluxes)]
pub struct Fluxes {
    pub <flux1>: f64,
    pub <flux2>: f64,
    pub streamflow: f64,
}

impl FluxesTimeseriesOps<Fluxes> for FluxesTimeseries {
    fn with_capacity(n: usize) -> Self { FluxesTimeseries::with_capacity(n) }
    fn push(&mut self, f: &Fluxes) { FluxesTimeseries::push(self, f); }
    fn len(&self) -> usize { FluxesTimeseries::len(self) }
    fn is_empty(&self) -> bool { FluxesTimeseries::is_empty(self) }
    fn with_len(n: usize) -> Self { FluxesTimeseries::with_len(n) }
    unsafe fn write_unchecked(&mut self, t: usize, f: &Fluxes) {
        unsafe { FluxesTimeseries::write_unchecked(self, t, f); }
    }
}
```

RULES: ALL fields MUST be `f64`. `streamflow` MUST always be present. Field order defines the canonical order for ALL other layers. The `FluxesTimeseriesOps` impl is identical for every model -- copy verbatim.

### 3.6 processes.rs -- `crates/pydrology-core/src/<model>/processes.rs`

```rust
/// <MODEL_NAME> core process functions.
use super::constants::<IMPORT_CONSTANTS>;

/// <STEP_DESCRIPTION>.
#[inline]
pub fn <process_name>(<arg1>: f64, <arg2>: f64, <param>: f64) -> (f64, f64) {
    // Pure math -- no mutation, no side effects
    (<output1>, <output2>)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tol: f64) {
        assert!((actual - expected).abs() < tol, "expected {expected} +/- {tol}, got {actual}");
    }

    #[test]
    fn <process_name>_conservation() {
        // Verify conservation laws and edge cases
    }
}
```

RULES: Every function MUST be `#[inline]` and `pub`. Return tuples. Import constants from `super::constants`, NEVER hardcode values.

### 3.7 run.rs -- `crates/pydrology-core/src/<model>/run.rs`

```rust
/// <MODEL_NAME> model orchestration.
use super::fluxes::{Fluxes, FluxesTimeseries};
use super::params::Parameters;
use super::processes;
use super::state::State;
use crate::traits::HydrologicalModel;

pub fn step(state: &State, params: &Parameters, precip: f64, pet: f64) -> (State, Fluxes) {
    // Call process functions, build new_state and fluxes
    let new_state = State { <store1>: <val>, <store2>: <val> };
    let fluxes = Fluxes { <flux1>: <val>, <flux2>: <val>, streamflow: <val> };
    (new_state, fluxes)
}

pub fn run(
    params: &Parameters, precip: &[f64], pet: &[f64], initial_state: Option<&State>,
) -> FluxesTimeseries {
    assert_eq!(precip.len(), pet.len(), "precip and pet must have the same length");
    let n = precip.len();
    let mut state = match initial_state {
        Some(s) => *s,
        None => State::initialize(params),
    };
    let mut outputs = FluxesTimeseries::with_len(n);
    for t in 0..n {
        let (new_state, fluxes) = step(&state, params, precip[t], pet[t]);
        unsafe { outputs.write_unchecked(t, &fluxes); }
        state = new_state;
    }
    outputs
}

#[derive(Debug, Clone, Copy)]
pub struct <Model>Forcing { pub precip: f64, pub pet: f64 }

pub struct <Model>;

impl HydrologicalModel for <Model> {
    const NAME: &'static str = "<MODEL_NAME>";
    type Params = Parameters;
    type State = State;
    type Forcing = <Model>Forcing;
    type Fluxes = Fluxes;
    type FluxesTimeseries = super::fluxes::FluxesTimeseries;
    type Context = ();

    fn prepare(_params: &Self::Params) -> Self::Context {}
    fn initialize_state(params: &Self::Params) -> Self::State { State::initialize(params) }
    fn step(
        state: &Self::State, params: &Self::Params, forcing: &Self::Forcing, _context: &Self::Context,
    ) -> (Self::State, Self::Fluxes) {
        step(state, params, forcing.precip, forcing.pet)
    }
}

pub fn run_from_slices(
    params: &Parameters, precip: &[f64], pet: &[f64], initial_state: Option<&State>,
) -> FluxesTimeseries {
    assert_eq!(precip.len(), pet.len(), "precip and pet must have the same length");
    let forcing: Vec<<Model>Forcing> = precip.iter().zip(pet)
        .map(|(&p, &e)| <Model>Forcing { precip: p, pet: e }).collect();
    <Model>::run(params, &forcing, initial_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_params() -> Parameters { Parameters::new(<val1>, <val2>) }

    #[test]
    fn step_returns_finite_values() {
        let p = test_params();
        let s = State::initialize(&p);
        let (ns, f) = step(&s, &p, <precip>, <pet>);
        assert!(ns.<store1>.is_finite());
        assert!(f.streamflow.is_finite());
    }

    #[test]
    fn run_output_length_matches_input() {
        let result = run(&test_params(), &[<v>; 5], &[<v>; 5], None);
        assert_eq!(result.len(), 5);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn run_panics_on_mismatched_lengths() {
        run(&test_params(), &[1.0, 2.0], &[1.0], None);
    }
}
```

RULES: `step()` takes `&State` (immutable), returns `(State, Fluxes)`. `run()` MUST use `with_len` + `write_unchecked` (not `with_capacity` + `push`). Use `*s` for Copy types, `s.clone()` for Clone-only.

### 3.8 lib.rs edit -- `crates/pydrology-core/src/lib.rs`

Add `pub mod <model>;` alphabetically:
```rust
pub mod cemaneige;
pub mod elevation;
pub mod forcing;
pub mod gr2m;
pub mod gr6j;
pub mod hbv_light;
pub mod <model>;    // <-- ADD alphabetically
pub mod metrics;
pub mod traits;
```

---

## 4. PyO3 Binding Template

**Create file: `crates/pydrology-python/src/<model>.rs`**

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::convert::{checked_slice, contiguous_slice};

use pydrology_core::<model>::params::Parameters;
use pydrology_core::<model>::run;
use pydrology_core::<model>::state::State;

define_timeseries_result! {
    /// <MODEL_NAME> run results.
    pub struct <Model>Result from pydrology_core::<model>::fluxes::FluxesTimeseries {
        <flux1>, <flux2>, streamflow,
    }
}

define_step_result! {
    /// <MODEL_NAME> single-timestep results.
    pub struct <Model>StepFluxes from pydrology_core::<model>::fluxes::Fluxes {
        <flux1>, <flux2>, streamflow,
    }
}

#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn <model>_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p_slice = checked_slice(&params, <N_PARAMS>, "params")?;
    let p = Parameters::new(p_slice[0], p_slice[1]);
    let precip_slice = contiguous_slice(&precip)?;
    let pet_slice = contiguous_slice(&pet)?;
    let state = match &initial_state {
        Some(s) => {
            let s_slice = checked_slice(s, <STATE_SIZE>, "initial_state")?;
            Some(State { <store1>: s_slice[0], <store2>: s_slice[1] })
        }
        None => None,
    };
    let result = run::run(&p, precip_slice, pet_slice, state.as_ref());
    let dict = timeseries_to_dict!(py, result, <flux1>, <flux2>, streamflow,);
    Ok(dict)
}

#[pyfunction]
fn <model>_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = checked_slice(&params, <N_PARAMS>, "params")?;
    let p = Parameters::new(p_slice[0], p_slice[1]);
    let s_slice = checked_slice(&state, <STATE_SIZE>, "state")?;
    let s = State { <store1>: s_slice[0], <store2>: s_slice[1] };
    let (new_state, fluxes) = run::step(&s, &p, precip, pet);
    let state_arr = PyArray1::from_vec(py, vec![new_state.<store1>, new_state.<store2>]);
    let dict = fluxes_to_dict!(py, fluxes, <flux1>, <flux2>, streamflow,);
    Ok((state_arr, dict))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "<model>")?;
    m.add_function(wrap_pyfunction!(<model>_run, &m)?)?;
    m.add_function(wrap_pyfunction!(<model>_step, &m)?)?;
    m.add_class::<<Model>Result>()?;
    m.add_class::<<Model>StepFluxes>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
```

### PyO3 lib.rs -- 3 edits to `crates/pydrology-python/src/lib.rs`

**Edit 1** -- module declaration (alphabetical among lines 5-8):
```rust
mod <model>;
```

**Edit 2** -- register call (alphabetical among lines 35-39):
```rust
    <model>::register(m)?;
```

**Edit 3** -- sys.modules loop (line 42, add to array):
```rust
    for name in &["cemaneige", "gr2m", "gr6j", "hbv_light", "<model>", "metrics"] {
```

---

## 5. Python Shim Template

**Create file: `python/pydrology/models/<model>.py`**

```python
"""<MODEL_NAME> rainfall-runoff model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from pydrology.types import Resolution

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_NAMES: tuple[str, ...] = ("<param1>", "<param2>")
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "<param1>": (<MIN>, <MAX>),
    "<param2>": (<MIN>, <MAX>),
}
STATE_SIZE: int = <N>
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameters:
    """<MODEL_NAME> calibrated parameters."""
    <param1>: float
    <param2>: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arr = np.array([self.<param1>, self.<param2>], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        if len(arr) != len(PARAM_NAMES):
            msg = f"Expected array of length {len(PARAM_NAMES)}, got {len(arr)}"
            raise ValueError(msg)
        return cls(<param1>=float(arr[0]), <param2>=float(arr[1]))


@dataclass
class State:
    """<MODEL_NAME> model state variables."""
    <store1>: float
    <store2>: float

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        return cls(
            <store1>=0.3 * params.<capacity_param>,
            <store2>=0.3 * params.<capacity_param>,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arr = np.array([self.<store1>, self.<store2>], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        return cls(<store1>=float(arr[0]), <store2>=float(arr[1]))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class <Model>Fluxes:
    """<MODEL_NAME> model flux outputs as arrays."""
    <flux1>: np.ndarray
    <flux2>: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ---------------------------------------------------------------------------
# Run / Step
# ---------------------------------------------------------------------------

def step(
    state: State, params: Parameters, precip: float, pet: float,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the <MODEL_NAME> model."""
    from pydrology._core import <model> as _rust

    new_state_arr, fluxes = _rust.<model>_step(
        np.ascontiguousarray(state, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        precip, pet,
    )
    new_state = State.from_array(new_state_arr)
    fluxes_converted: dict[str, float] = {k: float(v) for k, v in fluxes.items()}
    return new_state, fluxes_converted


def run(
    params: Parameters, forcing: ForcingData, initial_state: State | None = None,
) -> ModelOutput[<Model>Fluxes]:
    """Run the <MODEL_NAME> model over a timeseries."""
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"<MODEL_NAME> supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    from pydrology._core import <model> as _rust
    from pydrology.outputs import ModelOutput

    state_arr = np.ascontiguousarray(initial_state, dtype=np.float64) if initial_state is not None else None
    result = _rust.<model>_run(
        np.ascontiguousarray(params, dtype=np.float64),
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        state_arr,
    )
    return ModelOutput(
        time=forcing.time, fluxes=<Model>Fluxes(**result), snow=None, snow_layers=None,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_BOUNDS", "<Model>Fluxes", "PARAM_NAMES", "Parameters",
    "STATE_SIZE", "State", "SUPPORTED_RESOLUTIONS", "run", "step",
]

# Auto-register
import pydrology.models.<model> as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("<model>", _self)
```

### Python __init__.py edit -- `python/pydrology/__init__.py`

Add alphabetically among existing model imports:
```python
import pydrology.models.<model>  # noqa: F401 - triggers auto-registration
```

RULES:
- Rust import MUST be lazy: `from pydrology._core import <model> as _rust` inside function body.
- `np.ascontiguousarray()` for EVERY array passed to Rust. NEVER `np.asarray()`.
- `Parameters` MUST be `frozen=True`. `State` MUST NOT be frozen.
- `<Model>Fluxes` field order MUST match Rust `Fluxes` struct field order.

---

## 6. Cross-Layer Field Ordering Rules

Field ordering mismatches cause silent wrong results with no error.

### Flux field order -- 5 locations (canonical: Rust `Fluxes` struct)

| # | File | What |
|---|------|------|
| 1 | `crates/pydrology-core/src/<model>/fluxes.rs` | `Fluxes` struct field order (canonical) |
| 2 | `crates/pydrology-python/src/<model>.rs` | `define_timeseries_result!` field list |
| 3 | `crates/pydrology-python/src/<model>.rs` | `define_step_result!` field list |
| 4 | `crates/pydrology-python/src/<model>.rs` | `timeseries_to_dict!` and `fluxes_to_dict!` field lists |
| 5 | `python/pydrology/models/<model>.py` | `<Model>Fluxes` frozen dataclass field order |

### State layout -- 3 locations (canonical: Rust `State::to_vec()`)

| # | File | What |
|---|------|------|
| 1 | `crates/pydrology-core/src/<model>/state.rs` | `State` struct + `to_vec()`/`from_slice()` order |
| 2 | `crates/pydrology-python/src/<model>.rs` | State construction from `s_slice` indices |
| 3 | `python/pydrology/models/<model>.py` | `State.__array__()` and `State.from_array()` order |

### Parameter order -- 3 locations (canonical: Rust `PARAM_NAMES`)

| # | File | What |
|---|------|------|
| 1 | `crates/pydrology-core/src/<model>/constants.rs` | `PARAM_NAMES` order |
| 2 | `crates/pydrology-core/src/<model>/params.rs` | `Parameters` struct + `to_array()`/`from_array()` |
| 3 | `python/pydrology/models/<model>.py` | `PARAM_NAMES` + `Parameters.__array__()` + `from_array()` |

**RULE: When adding or reordering fields, update ALL locations. Copy from the Rust canonical source to all downstream locations.**

---

## 7. Testing Templates

### 7.1 test_types.py -- `tests/models/<model>/test_types.py`

```python
"""Tests for <MODEL_NAME> data types."""
import numpy as np
import pytest
from pydrology.models.<model> import Parameters, State


class TestParameters:
    def test_creates_with_valid_parameters(self) -> None:
        params = Parameters(<param1>=<val1>, <param2>=<val2>)
        assert params.<param1> == <val1>

    def test_is_frozen_dataclass(self) -> None:
        params = Parameters(<param1>=<val1>, <param2>=<val2>)
        with pytest.raises(AttributeError):
            params.<param1> = <other_val>  # type: ignore[misc]


class TestState:
    def test_initialize_creates_correct_fractions(self) -> None:
        state = State.initialize(Parameters(<param1>=<val1>, <param2>=<val2>))
        assert state.<store1> == pytest.approx(0.3 * <capacity_val>)

    def test_state_is_mutable(self) -> None:
        state = State.initialize(Parameters(<param1>=<val1>, <param2>=<val2>))
        state.<store1> = 200.0
        assert state.<store1> == 200.0


class TestStateArrayProtocol:
    def test_state_to_array_shape(self) -> None:
        state = State.initialize(Parameters(<param1>=<val1>, <param2>=<val2>))
        assert np.asarray(state).shape == (<STATE_SIZE>,)

    def test_state_roundtrip(self) -> None:
        original = State.initialize(Parameters(<param1>=<val1>, <param2>=<val2>))
        restored = State.from_array(np.asarray(original))
        assert restored.<store1> == original.<store1>


class TestParametersArrayProtocol:
    def test_params_to_array_shape(self) -> None:
        assert np.asarray(Parameters(<param1>=<val1>, <param2>=<val2>)).shape == (<N_PARAMS>,)

    def test_params_roundtrip(self) -> None:
        original = Parameters(<param1>=<val1>, <param2>=<val2>)
        restored = Parameters.from_array(np.asarray(original))
        assert restored.<param1> == original.<param1>
```

### 7.2 test_run.py -- `tests/models/<model>/test_run.py`

```python
"""Integration tests for <MODEL_NAME> run module."""
import numpy as np
import pandas as pd
import pytest
from pydrology import ForcingData, ModelOutput, Resolution
from pydrology.models.<model> import Parameters, State, run, step

EXPECTED_FLUX_KEYS = {"<flux1>", "<flux2>", "streamflow"}


@pytest.fixture
def typical_params() -> Parameters:
    return Parameters(<param1>=<val1>, <param2>=<val2>)

@pytest.fixture
def forcing() -> ForcingData:
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=<N>, freq="<FREQ>").values,
        precip=np.array([<values>]),
        pet=np.array([<values>]),
        resolution=Resolution.<resolution>,
    )

@pytest.fixture
def initialized_state(typical_params: Parameters) -> State:
    return State.initialize(typical_params)


class TestStep:
    def test_returns_new_state_and_fluxes(self, initialized_state: State, typical_params: Parameters) -> None:
        new_state, fluxes = step(state=initialized_state, params=typical_params, precip=<val>, pet=<val>)
        assert isinstance(new_state, State)
        assert isinstance(fluxes, dict)

    def test_fluxes_contains_all_expected_keys(self, initialized_state: State, typical_params: Parameters) -> None:
        _, fluxes = step(state=initialized_state, params=typical_params, precip=<val>, pet=<val>)
        assert set(fluxes.keys()) == EXPECTED_FLUX_KEYS

    def test_streamflow_is_non_negative(self, initialized_state: State, typical_params: Parameters) -> None:
        _, fluxes = step(state=initialized_state, params=typical_params, precip=<val>, pet=<val>)
        assert fluxes["streamflow"] >= 0.0


class TestRun:
    def test_returns_model_output(self, typical_params: Parameters, forcing: ForcingData) -> None:
        result = run(typical_params, forcing)
        assert isinstance(result, ModelOutput)
        assert set(result.fluxes.to_dict().keys()) == EXPECTED_FLUX_KEYS

    def test_output_length_matches_input(self, typical_params: Parameters, forcing: ForcingData) -> None:
        assert len(run(typical_params, forcing)) == len(forcing)

    def test_uses_provided_initial_state(self, typical_params: Parameters, forcing: ForcingData) -> None:
        custom = State(<store1>=<custom_val1>, <store2>=<custom_val2>)
        assert run(typical_params, forcing, custom).streamflow[0] != run(typical_params, forcing).streamflow[0]

    def test_streamflow_non_negative(self, typical_params: Parameters, forcing: ForcingData) -> None:
        assert (run(typical_params, forcing).fluxes.streamflow >= 0).all()


class TestResolutionValidation:
    def test_rejects_wrong_resolution(self, typical_params: Parameters) -> None:
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=6, freq="MS").values,
            precip=np.full(6, 50.0), pet=np.full(6, 30.0),
            resolution=Resolution.monthly,  # wrong for a daily model
        )
        with pytest.raises(ValueError, match="supports resolutions"):
            run(typical_params, forcing)


class TestValidationErrors:
    def test_run_rejects_wrong_length_params(self) -> None:
        from pydrology._core.<model> import <model>_run
        with pytest.raises(ValueError, match="must have"):
            <model>_run(np.array([<single_val>]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_step_rejects_wrong_length_state(self) -> None:
        from pydrology._core.<model> import <model>_step
        with pytest.raises(ValueError, match="must have"):
            <model>_step(np.array([<single_val>]), np.array([<val1>, <val2>]), 1.0, 1.0)
```

### 7.3 Conformance test edits -- `tests/test_model_conformance.py`

**Edit 1** -- `ALL_MODELS` list:
```python
ALL_MODELS: list[str] = ["gr2m", "gr6j", "hbv_light", "gr6j_cemaneige", "<model>"]
```

**Edit 2** -- Add `elif` branch to `_model_fixtures()`:
```python
    elif model_name == "<model>":
        n = <N_TIMESTEPS>
        time = pd.date_range("2020-01-01", periods=n, freq="<FREQ>")
        forcing = ForcingData(
            time=time.to_numpy(),
            precip=rng.uniform(<P_MIN>, <P_MAX>, size=n),
            pet=rng.uniform(<E_MIN>, <E_MAX>, size=n),
            resolution=Resolution.<resolution>,
        )
```

**Edit 3** -- Add `"<model>"` to `_make_initial_state()` (choose the right branch):
```python
    if model_name in ("gr2m", "gr6j", "<model>"):      # simple: State.initialize(params)
        return model.State.initialize(params)
```

**Edit 4** -- Add `"<model>"` to `_reconstruct_state()` (choose the right branch):
```python
    if model_name in ("gr2m", "gr6j", "<model>"):      # simple: State.from_array(arr)
        return model.State.from_array(arr)
```

### 7.4 Verification commands

Run in order. Every command MUST pass before the next.
This is the same protocol as `MODEL_CHECKLIST.md` Section 4.

```bash
# 1. Rust compiles and all Rust tests pass
cargo test --workspace

# 2. Clippy passes with no warnings
cargo clippy --workspace

# 3. Python extension builds
uv run maturin develop

# 4. Rust-level run binding import works
uv run python -c "from pydrology._core.<model> import <model>_run"

# 5. Rust-level step binding import works
uv run python -c "from pydrology._core.<model> import <model>_step"

# 6. Python shim run/step import works
uv run python -c "from pydrology.models.<model> import run, step"

# 7. Python shim types import works
uv run python -c "from pydrology.models.<model> import Parameters, State"

# 8. Model appears in registry
uv run python -c "from pydrology import registry; assert '<model>' in registry.list_models()"

# 9. Registry info is populated
uv run python -c "from pydrology import registry; info = registry.get_model_info('<model>'); print(info)"

# 10. Model-specific tests pass
uv run python -m pytest tests/models/<model>/ -v

# 11. Conformance suite passes
uv run python -m pytest tests/test_model_conformance.py -v

# 12. Registry tests pass
uv run python -m pytest tests/test_registry.py -v

# 13. Array contiguity tests pass
uv run python -m pytest tests/models/test_array_contiguity.py -v

# 14. Full test suite passes
uv run python -m pytest -x
```

---

## 8. Archetype Variations

### 8.1 Model with Context (like GR6J)

Diff from base template in `run.rs`:

```rust
pub struct <Model>Context {
    pub uh1: [f64; <SIZE1>],
    pub uh2: [f64; <SIZE2>],
}

// step() gains extra context parameters:
pub fn step(
    state: &State, params: &Parameters, precip: f64, pet: f64,
    uh1: &[f64; <SIZE1>], uh2: &[f64; <SIZE2>],
) -> (State, Fluxes) { /* ... */ }

// run() computes context once before the loop:
pub fn run(params: &Parameters, precip: &[f64], pet: &[f64], initial_state: Option<&State>) -> FluxesTimeseries {
    let (uh1, uh2) = compute_uh_ordinates(params.<uh_param>);
    // ... loop calls step(..., &uh1, &uh2)
}

// Trait impl uses Context:
impl HydrologicalModel for <Model> {
    type Context = <Model>Context;
    fn prepare(params: &Self::Params) -> Self::Context {
        let (uh1, uh2) = compute_uh_ordinates(params.<uh_param>);
        <Model>Context { uh1, uh2 }
    }
    fn step(state: &Self::State, params: &Self::Params, forcing: &Self::Forcing, context: &Self::Context)
        -> (Self::State, Self::Fluxes) {
        step(state, params, forcing.precip, forcing.pet, &context.uh1, &context.uh2)
    }
}
```

No PyO3 changes needed -- context is computed inside `run::run()`.

### 8.2 Variable-Size State (like HBV-Light)

Diff from base template:

```rust
// constants.rs -- function instead of constant:
pub fn compute_state_size(n_zones: usize) -> usize { <BASE> + n_zones * <PER_ZONE> }

// state.rs -- no Copy, Vec fields, extra init param:
#[derive(Debug, Clone)]  // NO Copy
pub struct State {
    pub <store>: f64,
    pub zone_stores: Vec<f64>,
}
impl State {
    pub fn initialize(params: &Parameters, n_zones: usize) -> Self { /* ... */ }
}
```

Python shim diff:
```python
@classmethod
def initialize(cls, params: Parameters, n_zones: int = 1) -> State: ...

@classmethod
def from_array(cls, arr: np.ndarray, n_zones: int = 1) -> State: ...
```

### 8.3 Standalone Model (like GR6J-CemaNeige)

Diff from base template:

- In `run.rs`: NO marker type, NO `HydrologicalModel` impl, NO `Forcing` struct.
- `step()` and `run()` accept extra arguments directly (e.g., `temp: &[f64]`, `catchment: &CatchmentProperties`).
- PyO3 `_run` function accepts extra `PyReadonlyArray1` arguments.
- Python `run()` takes extra kwargs (e.g., `catchment: Catchment`).
- Conformance test uses `run_kwargs` dict for extra arguments.
