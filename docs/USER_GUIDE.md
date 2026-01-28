# GR6J User Guide

A complete guide to using the GR6J hydrological model Python implementation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Input Data](#input-data)
3. [Model Parameters](#model-parameters)
4. [Running the Model](#running-the-model)
5. [Model Outputs](#model-outputs)
6. [Snow Module](#snow-module)
7. [Advanced Usage](#advanced-usage)
8. [Common Errors](#common-errors)

---

## Quick Start

### GR6J Only

The simplest way to run the model is with minimal forcing data and default parameters:

```python
import numpy as np
from gr6j import ForcingData, Parameters, run

# Create forcing data
n_days = 365
forcing = ForcingData(
    time=np.arange(n_days, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.random.exponential(5.0, n_days),  # [mm/day]
    pet=np.full(n_days, 3.5),                   # [mm/day]
)

# Define model parameters
params = Parameters(
    x1=350.0,   # Production store capacity [mm]
    x2=0.0,     # Intercatchment exchange coefficient [mm/day]
    x3=90.0,    # Routing store capacity [mm]
    x4=1.7,     # Unit hydrograph time constant [days]
    x5=0.0,     # Intercatchment exchange threshold [-]
    x6=5.0,     # Exponential store scale parameter [mm]
)

# Run the model
output = run(params, forcing)

# Access streamflow
print(output.gr6j.streamflow)  # numpy array of daily streamflow [mm/day]
```

### GR6J + CemaNeige Snow Module

For cold-climate catchments with snow influence:

```python
import numpy as np
from gr6j import Catchment, CemaNeige, ForcingData, Parameters, run

# Create forcing data with temperature
n_days = 365
forcing = ForcingData(
    time=np.arange(n_days, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.random.exponential(5.0, n_days),          # [mm/day]
    pet=np.full(n_days, 3.5),                           # [mm/day]
    temp=np.sin(np.linspace(0, 2*np.pi, n_days)) * 15,  # [deg C]
)

# Define catchment properties
catchment = Catchment(
    mean_annual_solid_precip=150.0,  # [mm/year]
)

# Define snow module parameters
snow = CemaNeige(
    ctg=0.97,  # Thermal state coefficient [-]
    kf=2.5,    # Degree-day melt factor [mm/deg C/day]
)

# Define GR6J parameters with snow module
params = Parameters(
    x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0,
    snow=snow,  # Attach snow module
)

# Run the model
output = run(params, forcing, catchment=catchment)

# Access outputs
print(output.gr6j.streamflow)   # [mm/day]
print(output.snow.snow_pack)    # [mm] snow water equivalent
print(output.snow.snow_melt)    # [mm/day]
```

---

## Input Data

### ForcingData

`ForcingData` is a validated container for time series forcing data. It uses Pydantic for automatic validation and type coercion.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `time` | `np.ndarray` (datetime64) | Yes | Timestamp for each observation |
| `precip` | `np.ndarray` (float64) | Yes | Precipitation [mm/day] |
| `pet` | `np.ndarray` (float64) | Yes | Potential evapotranspiration [mm/day] |
| `temp` | `np.ndarray` (float64) | No* | Temperature [deg C] |

*Required when snow module is enabled (`params.snow` is set).

**Validation Rules:**

- All arrays must be 1D
- All arrays must have the same length
- NaN values are rejected (fail fast design)
- Numeric arrays are automatically coerced to float64
- Time array is coerced to datetime64[ns]

**Example:**

```python
import numpy as np
from gr6j import ForcingData

# Basic creation
forcing = ForcingData(
    time=np.arange(365, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=precip_data,
    pet=pet_data,
)

# With temperature for snow module
forcing_with_temp = ForcingData(
    time=np.arange(365, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=precip_data,
    pet=pet_data,
    temp=temp_data,
)

# Check length
print(len(forcing))  # 365
```

### Catchment

Static catchment properties required for the snow module.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mean_annual_solid_precip` | `float` | Yes | Mean annual solid precipitation [mm/year] |
| `hypsometric_curve` | `np.ndarray` | No | Elevation distribution (future multi-layer support) |
| `input_elevation` | `float` | No | Elevation of forcing data [m] (future) |
| `n_layers` | `int` | No | Number of elevation bands (default: 1) |

**Example:**

```python
from gr6j import Catchment

# Basic catchment
catchment = Catchment(
    mean_annual_solid_precip=150.0,  # [mm/year]
)
```

**Typical Bounds:**

The `mean_annual_solid_precip` value should typically be in the range [0, 10000] mm/year. Values outside this range trigger a warning but are not rejected.

---

## Model Parameters

### Parameters (GR6J)

The GR6J model has 6 calibrated parameters:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **x1** | Production store capacity | mm | [1, 2500] |
| **x2** | Intercatchment exchange coefficient | mm/day | [-5, 5] |
| **x3** | Routing store capacity | mm | [1, 1000] |
| **x4** | Unit hydrograph time constant | days | [0.5, 10] |
| **x5** | Intercatchment exchange threshold | - | [-4, 4] |
| **x6** | Exponential store scale parameter | mm | [0.01, 20] |

**Physical Interpretation:**

- **x1**: Controls maximum soil moisture storage. Larger values mean more water retention before runoff.
- **x2**: Controls groundwater exchange with neighboring catchments. Positive = import, negative = export.
- **x3**: Controls baseflow recession characteristics through the routing store size.
- **x4**: Controls how quickly surface runoff reaches the outlet. Smaller = faster response.
- **x5**: Threshold controlling when groundwater exchange reverses direction.
- **x6**: Controls the exponential store contribution to slow baseflow (unique to GR6J).

**Example:**

```python
from gr6j import Parameters

params = Parameters(
    x1=350.0,
    x2=0.0,
    x3=90.0,
    x4=1.7,
    x5=0.0,
    x6=5.0,
)

# Access bounds for calibration
print(Parameters.BOUNDS)
# {'x1': (1.0, 2500.0), 'x2': (-5.0, 5.0), 'x3': (1.0, 1000.0), ...}
```

### CemaNeige (Snow Module)

The CemaNeige snow module has 2 calibrated parameters:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **ctg** | Thermal state weighting coefficient | - | [0, 1] |
| **kf** | Degree-day melt factor | mm/deg C/day | [1, 10] |

**Typical Calibrated Values:**

Based on validation across hundreds of catchments:

| Parameter | Typical Value | Common Range |
|-----------|---------------|--------------|
| ctg | 0.97 | 0.96 - 0.98 |
| kf | 2.5 | 2.2 - 2.8 |

**Physical Interpretation:**

- **ctg**: Controls thermal inertia of the snow pack. Values close to 1 mean slow temperature response; close to 0 means rapid adjustment to air temperature.
- **kf**: Controls how much snow melts per degree above 0 deg C. Higher values mean faster melt.

**Example:**

```python
from gr6j import CemaNeige

snow = CemaNeige(
    ctg=0.97,
    kf=2.5,
)

# Access bounds
print(CemaNeige.BOUNDS)
# {'ctg': (0.0, 1.0), 'kf': (0.0, 200.0)}
```

### has_snow Property

Check if snow module is enabled:

```python
from gr6j import CemaNeige, Parameters

# Without snow
params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
print(params.has_snow)  # False

# With snow
params_snow = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=CemaNeige(ctg=0.97, kf=2.5),
)
print(params_snow.has_snow)  # True
```

---

## Running the Model

### Function Signature

```python
def run(
    params: Parameters,
    forcing: ForcingData,
    catchment: Catchment | None = None,
    initial_state: State | None = None,
    initial_snow_state: CemaNeigeSingleLayerState | None = None,
) -> ModelOutput:
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `params` | `Parameters` | Yes | Model parameters (x1-x6) with optional snow |
| `forcing` | `ForcingData` | Yes | Input forcing data |
| `catchment` | `Catchment` | No* | Catchment properties |
| `initial_state` | `State` | No | Initial GR6J state |
| `initial_snow_state` | `CemaNeigeSingleLayerState` | No | Initial snow state |

*Required when snow module is enabled.

**Returns:**

`ModelOutput` containing GR6J outputs and optionally snow outputs.

### Examples

**Basic Run:**

```python
from gr6j import ForcingData, Parameters, run

params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
forcing = ForcingData(time=time_arr, precip=precip_arr, pet=pet_arr)

output = run(params, forcing)
```

**Run with Custom Initial State:**

```python
from gr6j import ForcingData, Parameters, State, run
import numpy as np

params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
forcing = ForcingData(time=time_arr, precip=precip_arr, pet=pet_arr)

# Custom initial state
initial_state = State(
    production_store=200.0,       # mm
    routing_store=50.0,           # mm
    exponential_store=0.0,        # mm
    uh1_states=np.zeros(20),
    uh2_states=np.zeros(40),
)

output = run(params, forcing, initial_state=initial_state)
```

**Run with Snow Module:**

```python
from gr6j import Catchment, CemaNeige, ForcingData, Parameters, run

params = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=CemaNeige(ctg=0.97, kf=2.5),
)
forcing = ForcingData(time=time_arr, precip=precip_arr, pet=pet_arr, temp=temp_arr)
catchment = Catchment(mean_annual_solid_precip=150.0)

output = run(params, forcing, catchment=catchment)
```

---

## Model Outputs

### ModelOutput

The `run()` function returns a `ModelOutput` object that provides access to all model results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | `np.ndarray` | Datetime array for each timestep |
| `gr6j` | `GR6JOutput` | GR6J model outputs |
| `snow` | `SnowOutput` or `None` | CemaNeige outputs (if snow enabled) |

**Methods:**

```python
# Get number of timesteps
len(output)  # 365

# Convert to pandas DataFrame
df = output.to_dataframe()
```

### GR6JOutput (20 fields)

All GR6J flux outputs as numpy arrays with the same length as input forcing.

| Field | Description | Unit |
|-------|-------------|------|
| `pet` | Potential evapotranspiration input | mm/day |
| `precip` | Precipitation input to GR6J* | mm/day |
| `production_store` | Production store level after timestep | mm |
| `net_rainfall` | Net rainfall after interception | mm/day |
| `storage_infiltration` | Water infiltrating to production store | mm/day |
| `actual_et` | Actual evapotranspiration | mm/day |
| `percolation` | Percolation from production store | mm/day |
| `effective_rainfall` | Total effective rainfall after percolation | mm/day |
| `q9` | Output from UH1 (90% branch) | mm/day |
| `q1` | Output from UH2 (10% branch) | mm/day |
| `routing_store` | Routing store level after timestep | mm |
| `exchange` | Groundwater exchange potential | mm/day |
| `actual_exchange_routing` | Actual exchange from routing store | mm/day |
| `actual_exchange_direct` | Actual exchange from direct branch | mm/day |
| `actual_exchange_total` | Total actual exchange | mm/day |
| `qr` | Outflow from routing store | mm/day |
| `qrexp` | Outflow from exponential store | mm/day |
| `exponential_store` | Exponential store level after timestep | mm |
| `qd` | Direct branch outflow | mm/day |
| `streamflow` | Total simulated streamflow | mm/day |

*When snow module is enabled, `precip` is the liquid water output from CemaNeige.

**Example:**

```python
output = run(params, forcing)

# Access individual arrays
streamflow = output.gr6j.streamflow
actual_et = output.gr6j.actual_et
routing_store = output.gr6j.routing_store

# Convert to dictionary
gr6j_dict = output.gr6j.to_dict()
```

### SnowOutput (12 fields)

CemaNeige snow module outputs (only present when snow module is enabled).

| Field | Description | Unit |
|-------|-------------|------|
| `precip_raw` | Original precipitation before snow processing | mm/day |
| `snow_pliq` | Liquid precipitation (rain) | mm/day |
| `snow_psol` | Solid precipitation (snow) | mm/day |
| `snow_pack` | Snow pack water equivalent after melt | mm |
| `snow_thermal_state` | Thermal state of snow pack | deg C |
| `snow_gratio` | Snow cover fraction after melt | - |
| `snow_pot_melt` | Potential melt | mm/day |
| `snow_melt` | Actual melt | mm/day |
| `snow_pliq_and_melt` | Total liquid output to GR6J | mm/day |
| `snow_temp` | Air temperature | deg C |
| `snow_gthreshold` | Melt threshold | mm |
| `snow_glocalmax` | Local maximum for hysteresis | mm |

**Example:**

```python
output = run(params, forcing, catchment=catchment)

if output.snow is not None:
    snow_pack = output.snow.snow_pack
    melt = output.snow.snow_melt

    # Convert to dictionary
    snow_dict = output.snow.to_dict()
```

### Converting to DataFrame

Convert all outputs to a pandas DataFrame with time as the index:

```python
output = run(params, forcing, catchment=catchment)

# Convert to DataFrame
df = output.to_dataframe()

# DataFrame has time as index, all outputs as columns
print(df.columns.tolist())
# ['pet', 'precip', 'production_store', ..., 'streamflow',
#  'precip_raw', 'snow_pliq', ...]  # snow columns if enabled

# Access by column name
df['streamflow'].plot()
df[['qr', 'qrexp', 'qd']].plot()
```

---

## Snow Module

### Enabling Snow

The snow module is enabled by attaching a `CemaNeige` object to the `Parameters`:

```python
from gr6j import CemaNeige, Parameters

# Create snow parameters
snow = CemaNeige(
    ctg=0.97,  # Thermal state coefficient [-]
    kf=2.5,    # Degree-day melt factor [mm/deg C/day]
)

# Attach to GR6J parameters
params = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=snow,
)
```

### Required Inputs

When snow is enabled, two additional inputs are required:

1. **Temperature data** in `ForcingData`:

```python
forcing = ForcingData(
    time=time_arr,
    precip=precip_arr,
    pet=pet_arr,
    temp=temp_arr,  # Required for snow!
)
```

2. **Catchment properties** with mean annual solid precipitation:

```python
from gr6j import Catchment

catchment = Catchment(
    mean_annual_solid_precip=150.0,  # [mm/year]
)

# Pass to run()
output = run(params, forcing, catchment=catchment)
```

### Snow Initialization

The snow state is automatically initialized if not provided. You can also provide custom initial snow state:

```python
from gr6j import CemaNeigeSingleLayerState

# Default initialization (recommended for fresh runs)
initial_snow = CemaNeigeSingleLayerState.initialize(
    mean_annual_solid_precip=150.0
)
# Creates: g=0 mm, etg=0 deg C, gthreshold=135 mm (0.9 * 150)

# Custom initialization (for warm-starting)
custom_snow_state = CemaNeigeSingleLayerState(
    g=50.0,           # Current snow pack [mm]
    etg=-2.0,         # Thermal state [deg C]
    gthreshold=135.0, # Melt threshold [mm]
    glocalmax=135.0,  # Local maximum [mm]
)

output = run(params, forcing, catchment=catchment, initial_snow_state=custom_snow_state)
```

### How Snow Module Works

The CemaNeige module processes precipitation before it enters GR6J:

1. **Precipitation partitioning**: Total precipitation is split into liquid (rain) and solid (snow) based on temperature
2. **Snow accumulation**: Solid precipitation adds to the snow pack
3. **Thermal state evolution**: Snow pack temperature tracks air temperature with inertia
4. **Melt calculation**: When conditions allow, snow melts using a degree-day approach
5. **Output**: Liquid precipitation + melt becomes the precipitation input to GR6J

When snow is enabled, the model outputs 32 columns (20 GR6J + 12 CemaNeige).

---

## Advanced Usage

### Custom Initial State

The GR6J model state can be initialized in two ways:

**Option 1: Derived from parameters (recommended for fresh runs)**

```python
from gr6j import Parameters, State

params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)

# Computes defaults as fractions of capacity:
#   production_store = 0.30 * X1 = 105 mm
#   routing_store    = 0.50 * X3 = 45 mm
#   exponential_store = 0 mm
initial_state = State.initialize(params)
```

**Option 2: Explicit values (for warm-starting from a previous run)**

```python
import numpy as np
from gr6j import State

# Values are direct mm amounts, independent of parameters
custom_state = State(
    production_store=200.0,      # mm
    routing_store=50.0,          # mm
    exponential_store=0.0,       # mm (can be negative)
    uh1_states=np.zeros(20),     # UH1 convolution states
    uh2_states=np.zeros(40),     # UH2 convolution states
)

output = run(params, forcing, initial_state=custom_state)
```

### Single Timestep Execution

For custom applications or debugging, you can execute the model one timestep at a time:

```python
from gr6j import Parameters, State, step
from gr6j.model.unit_hydrographs import compute_uh_ordinates

# Setup
params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
state = State.initialize(params)
uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

# Execute one timestep
new_state, fluxes = step(
    state=state,
    params=params,
    precip=10.0,  # mm/day
    pet=3.0,      # mm/day
    uh1_ordinates=uh1_ord,
    uh2_ordinates=uh2_ord,
)

print(f"Streamflow: {fluxes['streamflow']:.2f} mm/day")
print(f"New production store: {new_state.production_store:.2f} mm")

# Continue with next timestep
state = new_state
# ...
```

The `step()` function returns:
- `new_state`: Updated `State` object for the next timestep
- `fluxes`: Dictionary containing all 20 GR6J outputs for this timestep

### Warm-up Period

The model requires a **warm-up period** to initialize internal states before producing reliable outputs.

**Recommendations:**

| Aspect | Recommendation |
|--------|----------------|
| Duration | 365 days (1 year) minimum |
| Purpose | Allow stores to reach dynamic equilibrium |
| Cold climates | Use 2+ years for deep snow packs |
| Outputs | Discard warm-up period from analysis |

**Why warm-up is needed:**

- Initial store levels (production, routing, exponential) are typically set to default fractions
- These defaults rarely match actual catchment conditions
- The warm-up allows model states to "spin up" to realistic values

**Best practices:**

1. Use at least 1 year of data before the period of interest
2. For highly seasonal climates, use 2+ years of warm-up
3. Initialize states from a previous run if available (reduces warm-up needs)

```python
import numpy as np

# Load 3 years of data (1 year warm-up + 2 years analysis)
forcing = ForcingData(
    time=time_array,      # 3 years
    precip=precip_array,
    pet=pet_array,
)

output = run(params, forcing)

# Discard first year (warm-up)
warmup_days = 365
analysis_streamflow = output.gr6j.streamflow[warmup_days:]
analysis_time = output.time[warmup_days:]
```

### Chaining Simulations

You can warm-start a simulation using the final state from a previous run:

```python
# First run (calibration period)
output1 = run(params, forcing1)

# Extract final state from first run
# (Would need to extract from internal state - currently not directly exposed)
# This is a limitation - consider using step() for full control

# For now, run continuously
full_forcing = ForcingData(
    time=np.concatenate([forcing1.time, forcing2.time]),
    precip=np.concatenate([forcing1.precip, forcing2.precip]),
    pet=np.concatenate([forcing1.pet, forcing2.pet]),
)
output = run(params, full_forcing)
```

---

## Common Errors

### "forcing.temp required when snow module enabled"

**Cause:** You enabled snow (`params.snow` is set) but did not provide temperature data.

**Solution:** Add temperature to your ForcingData:

```python
forcing = ForcingData(
    time=time_arr,
    precip=precip_arr,
    pet=pet_arr,
    temp=temp_arr,  # Add this!
)
```

### "catchment required when snow module enabled"

**Cause:** You enabled snow but did not provide a Catchment object.

**Solution:** Create a Catchment with mean_annual_solid_precip:

```python
from gr6j import Catchment

catchment = Catchment(mean_annual_solid_precip=150.0)
output = run(params, forcing, catchment=catchment)  # Pass catchment!
```

### "precip array contains NaN values"

**Cause:** Your input data contains missing values.

**Solution:** Interpolate or fill NaN values before creating ForcingData:

```python
import numpy as np

# Option 1: Linear interpolation
precip = np.interp(
    np.arange(len(precip_raw)),
    np.where(~np.isnan(precip_raw))[0],
    precip_raw[~np.isnan(precip_raw)],
)

# Option 2: Fill with zero (for precipitation)
precip = np.nan_to_num(precip_raw, nan=0.0)

# Option 3: Forward fill
import pandas as pd
precip = pd.Series(precip_raw).ffill().bfill().values
```

### "precip length X does not match time length Y"

**Cause:** Your arrays have different lengths.

**Solution:** Ensure all arrays have the same number of elements:

```python
# Check lengths before creating ForcingData
print(f"time: {len(time_arr)}")
print(f"precip: {len(precip_arr)}")
print(f"pet: {len(pet_arr)}")

# Ensure they match
assert len(time_arr) == len(precip_arr) == len(pet_arr)
```

### "precip array must be 1D, got 2D"

**Cause:** Your array has the wrong shape.

**Solution:** Flatten or reshape your array:

```python
# Flatten if needed
precip = precip_arr.flatten()

# Or squeeze out singleton dimensions
precip = np.squeeze(precip_arr)
```

### "pet array must be 1D, got 2D"

Same solution as above, applies to all input arrays.

### Parameter out of range warnings

**Cause:** A parameter value is outside its typical calibration range.

**Note:** These are warnings, not errors. The model will still run. Out-of-range values may be valid for specific catchments or research purposes.

```python
import logging

# Suppress warnings if intentional
logging.getLogger("gr6j").setLevel(logging.ERROR)

# Or check bounds before creating Parameters
x1_value = 3000.0
if x1_value < 1.0 or x1_value > 2500.0:
    print(f"Warning: x1={x1_value} is outside typical range [1, 2500]")
```

---

## Further Reading

- [Model equations and algorithm](MODEL_DEFINITION.md)
- [Snow module technical details](CEMANEIGE.md)
- [Forcing data contract](FORCING_DATA_CONTRACT.md)

## References

- Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T. and Andreassian, V. (2011). A downward structural sensitivity analysis of hydrological models to improve low-flow simulation. *Journal of Hydrology*, 411(1-2), 66-76.

- Valery, A., Andreassian, V., and Perrin, C. (2014). 'As simple as possible but not simpler': What is useful in a temperature-based snow-accounting routine? Part 1 - Comparison of six snow accounting routines on 380 catchments. *Journal of Hydrology*, 517, 1166-1175.
