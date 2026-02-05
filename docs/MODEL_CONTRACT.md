# Model Interface Contract

This document describes the model interface contract that all models in PyDrology must implement to be compatible with the model registry and calibration system.

## Overview

PyDrology uses a model registry (`pydrology.registry`) to enable dynamic model discovery and runtime model selection. Each registered model must export a specific set of symbols that define its parameters, state, bounds, and execution functions.

## Required Exports

Every model module must export the following:

| Export | Type | Description |
|--------|------|-------------|
| `PARAM_NAMES` | `tuple[str, ...]` | Parameter names in order (e.g., `("x1", "x2", ...)`) |
| `DEFAULT_BOUNDS` | `dict[str, tuple[float, float]]` | Parameter bounds: `{name: (min, max)}` |
| `STATE_SIZE` | `int` | Number of elements in state array representation |
| `SUPPORTED_RESOLUTIONS` | `tuple[Resolution, ...]` | Supported temporal resolutions for forcing data |
| `Parameters` | `class` | Frozen dataclass for model parameters |
| `State` | `class` | Mutable dataclass for model state |
| `run` | `function` | Execute model over a timeseries |
| `step` | `function` | Execute a single timestep |

### Optional Exports

| Export | Type | Default | Description |
|--------|------|---------|-------------|
| `HAS_NUMBA` | `bool` | `True` | Whether Numba-optimized kernels are available |

## Parameters Class Requirements

The `Parameters` class must be a frozen dataclass with:

1. **`from_array(arr: np.ndarray) -> Parameters`**: Class method to reconstruct from array
2. **`__array__(dtype=None) -> np.ndarray`**: Convert to 1D array for Numba

**Example:**

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Parameters:
    """Model parameters."""
    x1: float  # Parameter 1
    x2: float  # Parameter 2

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to 1D array."""
        arr = np.array([self.x1, self.x2], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Parameters":
        """Reconstruct Parameters from array."""
        return cls(x1=float(arr[0]), x2=float(arr[1]))
```

## State Class Requirements

The `State` class must be a mutable dataclass with:

1. **`from_array(arr: np.ndarray) -> State`**: Class method to reconstruct from array
2. **`__array__(dtype=None) -> np.ndarray`**: Convert to 1D array for Numba
3. **`initialize(params: Parameters) -> State`**: (Optional) Class method for default initialization

**Example:**

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    """Model state variables."""
    store1: float  # Store 1 level [mm]
    store2: float  # Store 2 level [mm]

    @classmethod
    def initialize(cls, params: Parameters) -> "State":
        """Create initial state from parameters."""
        return cls(
            store1=0.3 * params.x1,  # 30% of capacity
            store2=0.5 * params.x2,  # 50% of capacity
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to 1D array."""
        arr = np.array([self.store1, self.store2], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        """Reconstruct State from array."""
        return cls(store1=float(arr[0]), store2=float(arr[1]))
```

## Run Function Requirements

The `run()` function executes the model over a full timeseries.

**Signature:**

```python
def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
    **kwargs,  # Model-specific arguments (e.g., catchment for snow models)
) -> ModelOutput:
    ...
```

**Requirements:**

1. Must accept `params`, `forcing`, and optional `initial_state`
2. Must return a `ModelOutput` compatible object
3. Should use Numba-optimized kernels when available
4. Should handle state initialization if `initial_state` is None

## Step Function Requirements

The `step()` function executes a single model timestep.

**Signature:**

```python
def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    **kwargs,  # Model-specific arguments
) -> tuple[State, dict[str, float]]:
    ...
```

**Requirements:**

1. Must accept current `state`, `params`, and scalar forcing values
2. Must return `(new_state, fluxes)` tuple
3. `new_state` is the updated State object
4. `fluxes` is a dictionary of output variables for this timestep

## Constants

### PARAM_NAMES

Tuple of parameter names in the order they appear in the array representation.

```python
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3")
```

### DEFAULT_BOUNDS

Dictionary mapping parameter names to `(min, max)` bounds for calibration.

```python
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
}
```

### STATE_SIZE

Total number of elements in the state array representation.

```python
STATE_SIZE: int = 63  # For GR6J: 3 stores + 20 UH1 + 40 UH2
```

### SUPPORTED_RESOLUTIONS

Tuple of `Resolution` enum values indicating which temporal resolutions the model supports.

```python
from pydrology.types import Resolution

SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)
```

Models validate that the forcing data resolution is in `SUPPORTED_RESOLUTIONS` before running. If a model supports multiple resolutions:

```python
# Model that supports both daily and hourly data
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.hourly, Resolution.daily)
```

The registry validates that all registered models export `SUPPORTED_RESOLUTIONS` and that it is a non-empty tuple of `Resolution` values.

**Current Model Support:**

| Model | Supported Resolutions |
|-------|----------------------|
| `gr6j` | daily |
| `gr6j_cemaneige` | daily |
| `hbv_light` | daily |
| `gr2m` | `(Resolution.monthly,)` |

## Registration

Models auto-register themselves when imported. The registration code should be at the module level:

```python
# At the end of __init__.py
import my_model as _self
from pydrology.registry import register

register("my_model", _self)
```

## Example: Minimal Model Implementation

Here's a complete minimal model implementation:

```python
# my_model/__init__.py
"""Minimal model example."""

from .constants import DEFAULT_BOUNDS, PARAM_NAMES, STATE_SIZE, SUPPORTED_RESOLUTIONS
from .run import run, step
from .types import Parameters, State

__all__ = [
    "DEFAULT_BOUNDS",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "SUPPORTED_RESOLUTIONS",
    "State",
    "run",
    "step",
]

# Auto-register
import my_model as _self
from pydrology.registry import register

register("my_model", _self)
```

```python
# my_model/constants.py
"""Model constants."""

from pydrology.types import Resolution

PARAM_NAMES: tuple[str, ...] = ("capacity", "coefficient")

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "capacity": (10.0, 1000.0),
    "coefficient": (0.01, 1.0),
}

STATE_SIZE: int = 1  # Single store

SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)
```

```python
# my_model/types.py
"""Model data types."""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Parameters:
    """Model parameters."""
    capacity: float    # Store capacity [mm]
    coefficient: float  # Outflow coefficient [-]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arr = np.array([self.capacity, self.coefficient], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Parameters":
        return cls(capacity=float(arr[0]), coefficient=float(arr[1]))


@dataclass
class State:
    """Model state."""
    store: float  # Store level [mm]

    @classmethod
    def initialize(cls, params: Parameters) -> "State":
        return cls(store=0.3 * params.capacity)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arr = np.array([self.store], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        return cls(store=float(arr[0]))
```

```python
# my_model/run.py
"""Model execution functions."""

import numpy as np
from pydrology.types import ForcingData
from .types import Parameters, State


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
) -> tuple[State, dict[str, float]]:
    """Execute single timestep."""
    # Simple bucket model
    store = state.store + precip

    # Limit to capacity
    overflow = max(0, store - params.capacity)
    store = min(store, params.capacity)

    # Evaporation
    et = min(pet, store)
    store = store - et

    # Baseflow
    baseflow = params.coefficient * store
    store = store - baseflow

    # Total outflow
    streamflow = overflow + baseflow

    new_state = State(store=store)
    fluxes = {
        "precip": precip,
        "pet": pet,
        "store": store,
        "actual_et": et,
        "overflow": overflow,
        "baseflow": baseflow,
        "streamflow": streamflow,
    }

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
) -> dict:
    """Execute model over timeseries."""
    if initial_state is None:
        initial_state = State.initialize(params)

    n = len(forcing)
    outputs = {key: np.zeros(n) for key in ["precip", "pet", "store",
               "actual_et", "overflow", "baseflow", "streamflow"]}

    state = initial_state
    for i in range(n):
        state, fluxes = step(state, params, forcing.precip[i], forcing.pet[i])
        for key, value in fluxes.items():
            outputs[key][i] = value

    return outputs
```

## Using Registered Models

Once registered, models can be accessed via the registry:

```python
from pydrology import get_model, list_models, get_model_info

# List available models
print(list_models())  # ['gr6j', 'gr6j_cemaneige', 'my_model']

# Get model info
info = get_model_info("my_model")
print(info["param_names"])  # ('capacity', 'coefficient')

# Get model module and use it
model = get_model("my_model")
params = model.Parameters(capacity=500.0, coefficient=0.1)
output = model.run(params, forcing)
```

## Calibration Compatibility

Models registered with the registry are automatically compatible with `calibrate()`:

```python
from pydrology import calibrate

result = calibrate(
    model="my_model",  # Use registered name
    forcing=forcing,
    observed=observed,
    objectives=["nse"],
    use_default_bounds=True,  # Uses DEFAULT_BOUNDS from model
    warmup=365,
)
```

## See Also

- [User Guide](USER_GUIDE.md) - General usage documentation
- [GR6J Model Definition](MODEL_DEFINITION.md) - GR6J model equations
- [CemaNeige](CEMANEIGE.md) - Snow module documentation
