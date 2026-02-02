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
8. [Utilities](#utilities)
9. [Calibration](#calibration)
10. [Common Errors](#common-errors)

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
| `hypsometric_curve` | `np.ndarray` | No* | 101-point elevation distribution [m]. Use `analyze_dem()` to compute from a DEM. |
| `input_elevation` | `float` | No* | Elevation of forcing data [m] |
| `n_layers` | `int` | No | Number of elevation bands (default: 1) |
| `temp_gradient` | `float` | No | Temperature lapse rate [deg C/100m]. Default: 0.6 |
| `precip_gradient` | `float` | No | Precipitation gradient [m^-1]. Default: 0.00041 |

*Required when `n_layers > 1`.

**Example:**

```python
from gr6j import Catchment

# Basic catchment (single layer)
catchment = Catchment(
    mean_annual_solid_precip=150.0,  # [mm/year]
)

# Multi-layer catchment (5 elevation bands)
import numpy as np
catchment_ml = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,
    hypsometric_curve=np.linspace(200.0, 2000.0, 101),
    input_elevation=500.0,
)
```

---

## Model Parameters

### Parameters (GR6J)

The GR6J model has 6 calibrated parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| **x1** | Production store capacity | mm |
| **x2** | Intercatchment exchange coefficient | mm/day |
| **x3** | Routing store capacity | mm |
| **x4** | Unit hydrograph time constant | days |
| **x5** | Intercatchment exchange threshold | - |
| **x6** | Exponential store scale parameter | mm |

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
```

### CemaNeige (Snow Module)

The CemaNeige snow module has 2 calibrated parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| **ctg** | Thermal state weighting coefficient | - |
| **kf** | Degree-day melt factor | mm/deg C/day |

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
    initial_snow_state: CemaNeigeSingleLayerState | CemaNeigeMultiLayerState | None = None,
) -> ModelOutput:
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `params` | `Parameters` | Yes | Model parameters (x1-x6) with optional snow |
| `forcing` | `ForcingData` | Yes | Input forcing data |
| `catchment` | `Catchment` | No* | Catchment properties |
| `initial_state` | `State` | No | Initial GR6J state |
| `initial_snow_state` | `CemaNeigeSingleLayerState \| CemaNeigeMultiLayerState` | No | Initial snow state |

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
| `snow_layers` | `SnowLayerOutputs` or `None` | Per-layer snow outputs (if multi-layer enabled) |

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

### SnowLayerOutputs (Multi-Layer)

Per-layer snow outputs for multi-layer simulations. Contains 2D arrays with shape `(n_timesteps, n_layers)`.

| Field | Description | Shape |
|-------|-------------|-------|
| `layer_elevations` | Representative elevation of each layer [m] | (n_layers,) |
| `layer_fractions` | Area fraction of each layer [-] | (n_layers,) |
| `snow_pack` | Snow pack per layer [mm] | (n_timesteps, n_layers) |
| `snow_thermal_state` | Thermal state per layer [deg C] | (n_timesteps, n_layers) |
| `snow_gratio` | Snow cover fraction per layer [-] | (n_timesteps, n_layers) |
| `snow_melt` | Actual melt per layer [mm/day] | (n_timesteps, n_layers) |
| `snow_pliq_and_melt` | Total liquid output per layer [mm/day] | (n_timesteps, n_layers) |
| `layer_temp` | Extrapolated temperature per layer [deg C] | (n_timesteps, n_layers) |
| `layer_precip` | Extrapolated precipitation per layer [mm/day] | (n_timesteps, n_layers) |

**Example:**

```python
output = run(params, forcing, catchment=catchment_ml)

if output.snow_layers is not None:
    # Per-layer data
    layer_snow = output.snow_layers.snow_pack       # (n_timesteps, 5)
    layer_temps = output.snow_layers.layer_temp      # (n_timesteps, 5)
    elevations = output.snow_layers.layer_elevations  # (5,)
    print(f"Layer elevations: {elevations}")
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

1. **Catchment properties** with mean annual solid precipitation:

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

### Multi-Layer Snow Simulation

For better representation of snow processes in mountainous catchments, CemaNeige supports multiple elevation bands:

```python
import numpy as np
from gr6j import Catchment, CemaNeige, CemaNeigeMultiLayerState, ForcingData, Parameters, run

# Define multi-layer catchment
catchment = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,
    hypsometric_curve=np.linspace(200.0, 2000.0, 101),
    input_elevation=500.0,
)

params = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=CemaNeige(ctg=0.97, kf=2.5),
)

forcing = ForcingData(
    time=time_arr,
    precip=precip_arr,
    pet=pet_arr,
    temp=temp_arr,
)

output = run(params, forcing, catchment=catchment)

# Aggregated output (area-weighted average across layers)
print(output.snow.snow_pack)  # 1D array

# Per-layer output
print(output.snow_layers.snow_pack)   # 2D array: (n_timesteps, 5)
print(output.snow_layers.layer_temp)  # 2D array: (n_timesteps, 5)
```

Each layer runs an independent CemaNeige instance with temperature and precipitation extrapolated using elevation gradients. The aggregated output (`output.snow`) is the area-weighted average, while `output.snow_layers` provides per-layer detail.

**Custom initial state:**

```python
from gr6j import CemaNeigeMultiLayerState

# Initialize multi-layer snow state
initial_state = CemaNeigeMultiLayerState.initialize(
    n_layers=5,
    mean_annual_solid_precip=150.0,
)

output = run(params, forcing, catchment=catchment, initial_snow_state=initial_state)
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

## Utilities

### DEM Analysis

The `analyze_dem()` utility computes elevation statistics from a basin-clipped DEM raster file. This is particularly useful for setting up multi-layer CemaNeige simulations.

**Basic usage:**

```python
from gr6j.utils import analyze_dem

dem = analyze_dem("data/my_basin_dem.tif")
print(dem)
# DEMStatistics(
#   min_elevation=452.30,
#   max_elevation=2147.80,
#   mean_elevation=1203.50,
#   median_elevation=1156.20,
#   hypsometric_curve=<array shape=(101,)>
# )
```

**Using with CemaNeige multi-layer:**

```python
from gr6j import Catchment, CemaNeige, Parameters
from gr6j.utils import analyze_dem

# Analyze the DEM
dem = analyze_dem("data/basin_dem.tif")

# Configure multi-layer catchment using DEM statistics
catchment = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,
    hypsometric_curve=dem.hypsometric_curve,
    input_elevation=dem.median_elevation,  # Or use station elevation if known
)

params = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=CemaNeige(ctg=0.97, kf=2.5),
)
```

**DEMStatistics attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `min_elevation` | `float` | Minimum elevation in the DEM [m] |
| `max_elevation` | `float` | Maximum elevation in the DEM [m] |
| `mean_elevation` | `float` | Mean elevation [m] |
| `median_elevation` | `float` | Median elevation [m] |
| `hypsometric_curve` | `np.ndarray` | 101-point elevation percentiles (0-100%) [m] |

**Choosing `input_elevation`:**

The `input_elevation` parameter for `Catchment` should be the elevation where your forcing data (precipitation, temperature) was measured. Options:

1. **Weather station elevation** (recommended) - If you know where your forcing data comes from
2. **Median elevation** (`dem.median_elevation`) - A reasonable default if station elevation is unknown
3. **Mean elevation** (`dem.mean_elevation`) - Alternative default

**Supported formats:**

- GeoTIFF (`.tif`, `.tiff`) - recommended
- Any raster format supported by GDAL/rasterio

---

### Solid Precipitation Utilities

The solid precipitation utilities compute the fraction and amount of precipitation falling as snow using the USACE (US Army Corps of Engineers) formula. These are useful for:

- Estimating `mean_annual_solid_precip` for CemaNeige configuration
- Analyzing snow climatology for a basin
- Pre-processing forcing data for snow-dominated catchments

**USACE Formula:**

The solid fraction is computed as a linear function of temperature:

$$f_{solid} = \begin{cases}
1 & \text{if } T \leq T_{snow} \\
\frac{T_{rain} - T}{T_{rain} - T_{snow}} & \text{if } T_{snow} < T < T_{rain} \\
0 & \text{if } T \geq T_{rain}
\end{cases}$$

With default thresholds $T_{snow} = -1°C$ and $T_{rain} = 3°C$.

#### compute_solid_fraction

Compute the fraction of precipitation falling as snow.

```python
from gr6j.utils import compute_solid_fraction
import numpy as np

temp = np.array([-5.0, 0.0, 1.0, 5.0])  # °C
fraction = compute_solid_fraction(temp)
print(fraction)  # [1.0, 0.75, 0.5, 0.0]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `temp` | `np.ndarray` | required | Temperature array [°C] |
| `t_snow` | `float` | -1.0 | All-snow threshold [°C] |
| `t_rain` | `float` | 3.0 | All-rain threshold [°C] |

**Returns:** `np.ndarray` - Solid fraction values in [0, 1]

#### compute_solid_precip

Compute solid precipitation from total precipitation and temperature.

```python
from gr6j.utils import compute_solid_precip
import numpy as np

precip = np.array([10.0, 10.0, 10.0, 10.0])  # mm/day
temp = np.array([-5.0, 0.0, 1.0, 5.0])       # °C
solid = compute_solid_precip(precip, temp)
print(solid)  # [10.0, 7.5, 5.0, 0.0]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `precip` | `np.ndarray` | required | Total precipitation [mm/day] |
| `temp` | `np.ndarray` | required | Temperature [°C] |
| `t_snow` | `float` | -1.0 | All-snow threshold [°C] |
| `t_rain` | `float` | 3.0 | All-rain threshold [°C] |

**Returns:** `np.ndarray` - Solid precipitation [mm/day]

#### compute_mean_annual_solid_precip

Compute mean annual solid precipitation for CemaNeige configuration.

```python
from gr6j.utils import compute_mean_annual_solid_precip
import numpy as np

# Load your forcing data (example with 3 years of daily data)
precip = np.random.uniform(0, 20, 365 * 3)  # mm/day
temp = np.random.uniform(-10, 25, 365 * 3)  # °C

mean_annual = compute_mean_annual_solid_precip(precip, temp)
print(f"Mean annual solid precip: {mean_annual:.1f} mm/year")
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `precip` | `np.ndarray` | required | Daily precipitation [mm/day] |
| `temp` | `np.ndarray` | required | Daily temperature [°C] |
| `t_snow` | `float` | -1.0 | All-snow threshold [°C] |
| `t_rain` | `float` | 3.0 | All-rain threshold [°C] |

**Returns:** `float` - Mean annual solid precipitation [mm/year]

**Integration with Catchment:**

```python
from gr6j import Catchment, CemaNeige, ForcingData, Parameters, run
from gr6j.utils import compute_mean_annual_solid_precip

# Compute from historical forcing data
mean_annual_solid = compute_mean_annual_solid_precip(
    precip=historical_precip,
    temp=historical_temp,
)

# Use in Catchment configuration
catchment = Catchment(mean_annual_solid_precip=mean_annual_solid)

params = Parameters(
    x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5,
    snow=CemaNeige(ctg=0.97, kf=2.5),
)

output = run(params, forcing, catchment=catchment)
```

---

## Calibration

The GR6J package includes automatic parameter calibration using evolutionary algorithms via the [ctrl-freak](https://github.com/hydrosolutions/ctrl-freak) library.

### Single-Objective Calibration

Optimize parameters to maximize a single metric (e.g., Nash-Sutcliffe Efficiency):

```python
import numpy as np
from gr6j import ForcingData, calibrate, ObservedData, Parameters

# Prepare forcing data (including warmup period)
n_days = 365 + 365  # 1 year warmup + 1 year calibration
forcing = ForcingData(
    time=np.datetime64("2019-01-01") + np.arange(n_days),
    precip=precip_data,
    pet=pet_data,
)

# Observed streamflow (post-warmup only)
observed = ObservedData(
    time=forcing.time[365:],
    streamflow=observed_streamflow,
)

# Define parameter bounds
bounds = {
    "x1": (1, 2500),    # Production store capacity [mm]
    "x2": (-5, 5),      # Intercatchment exchange [mm/day]
    "x3": (1, 1000),    # Routing store capacity [mm]
    "x4": (0.5, 10),    # UH time constant [days]
    "x5": (-4, 4),      # Exchange threshold [-]
    "x6": (0.01, 20),   # Exponential store parameter [mm]
}

# Run calibration
result = calibrate(
    forcing=forcing,
    observed=observed,
    objectives=["nse"],
    bounds=bounds,
    warmup=365,
    population_size=50,
    generations=100,
    seed=42,
)

print(f"Best NSE: {result.score['nse']:.3f}")
print(f"X1: {result.parameters.x1:.1f}")
```

By default, calibration displays a progress bar showing the current generation and best metric value. To disable the progress bar (e.g., for batch processing or logging):

```python
result = calibrate(
    ...,
    progress=False,  # Disable progress bar
)
```

### Multi-Objective Calibration

Optimize for multiple metrics simultaneously to obtain a Pareto front:

```python
from gr6j import Catchment, ForcingData, ObservedData, calibrate

# With snow module enabled
forcing = ForcingData(
    time=np.datetime64("2019-01-01") + np.arange(n_days),
    precip=precip_data,
    pet=pet_data,
    temp=temp_data,
)

catchment = Catchment(mean_annual_solid_precip=150.0)

# Bounds including snow parameters
bounds = {
    "x1": (1, 2500), "x2": (-5, 5), "x3": (1, 1000),
    "x4": (0.5, 10), "x5": (-4, 4), "x6": (0.01, 20),
    "ctg": (0, 1),      # Thermal state coefficient [-]
    "kf": (1, 10),      # Degree-day melt factor [mm/°C/day]
}

# Multi-objective returns list of Pareto-optimal solutions
solutions = calibrate(
    forcing=forcing,
    observed=observed,
    objectives=["nse", "log_nse"],
    bounds=bounds,
    catchment=catchment,
    snow=True,
    warmup=365,
    population_size=50,
    generations=100,
)

print(f"Found {len(solutions)} Pareto-optimal solutions")
for i, sol in enumerate(solutions[:3]):
    print(f"  {i+1}: NSE={sol.score['nse']:.3f}, log-NSE={sol.score['log_nse']:.3f}")
```

For multi-objective optimization, the progress bar shows the current Pareto front size instead of a single best value.

### ObservedData

Container for observed streamflow matching the post-warmup period of forcing data.

| Field | Type | Description |
|-------|------|-------------|
| `time` | `np.ndarray` (datetime64) | Timestamp for each observation |
| `streamflow` | `np.ndarray` (float64) | Observed streamflow [mm/day] |

**Validation:** Same rules as `ForcingData` - 1D arrays, no NaN, matching lengths.

### Available Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| `nse` | maximize | Nash-Sutcliffe Efficiency |
| `log_nse` | maximize | NSE on log-transformed flows (emphasizes low flows) |
| `kge` | maximize | Kling-Gupta Efficiency |
| `pbias` | minimize | Percent Bias |
| `rmse` | minimize | Root Mean Square Error |
| `mae` | minimize | Mean Absolute Error |

List available metrics programmatically:

```python
from gr6j import list_metrics
print(list_metrics())  # ['kge', 'log_nse', 'mae', 'nse', 'pbias', 'rmse']
```

### Custom Metrics

Register custom metrics using the `@register` decorator:

```python
from gr6j.calibration.metrics import register
import numpy as np

@register("minimize")
def custom_rmse(observed, simulated):
    """Custom RMSE focusing on high flows."""
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    # Weight high flows more heavily
    weights = obs / obs.max()
    return float(np.sqrt(np.mean(weights * (obs - sim) ** 2)))
```

### Warmup Period

The warmup period allows model stores to reach dynamic equilibrium before evaluation:

| Climate | Recommended Warmup |
|---------|-------------------|
| Temperate | 365 days (1 year) |
| Cold/snow-dominated | 730 days (2 years) |
| Highly seasonal | 730+ days |

**Important:** `len(observed) == len(forcing) - warmup` must hold.

**Note:** The `observed` data must contain **only** the post-warmup period. If your observed data includes the warmup period (e.g., same length as forcing), you must slice it before passing to `calibrate()`:

```python
warmup = 365

# If observed covers the full period (including warmup), slice it:
observed = ObservedData(
    time=full_observed_time[warmup:],
    streamflow=full_observed_streamflow[warmup:],
)

result = calibrate(
    forcing=forcing,       # Full period (warmup + calibration)
    observed=observed,     # Post-warmup only
    warmup=warmup,
    ...
)
```

Passing observed data that includes the warmup period will raise a `ValueError`.

### Progress Monitoring

By default, `calibrate()` displays a tqdm progress bar that shows:
- **Single-objective (GA)**: Current generation and best metric value
- **Multi-objective (NSGA-II)**: Current generation and Pareto front size

Example output:
```
Calibrating: 45%|████████████              | 45/100 [00:30<00:37, 1.50gen/s, best=0.8523]
```

To disable the progress bar for batch processing or when using custom logging:
```python
result = calibrate(..., progress=False)
```

### Parallel Execution

Speed up calibration by evaluating individuals in parallel:

```python
result = calibrate(
    forcing=forcing,
    observed=observed,
    objectives=["nse"],
    bounds=bounds,
    warmup=365,
    n_workers=-1,  # Use all CPU cores
)
```

| Value | Behavior |
|-------|----------|
| `1` | Sequential execution (default) |
| `-1` | Use all available CPU cores |
| `N` | Use N parallel workers |

**Note:** Parallel execution spawns separate processes. Logging configuration
from the main process (e.g., suppressing parameter warnings) does not propagate
to worker processes.

### Early Stopping via Callback

Custom callbacks for monitoring or early stopping are only available when `progress=False`:

```python
def my_callback(result, generation):
    # For GA: result.best returns (params, fitness)
    _, fitness = result.best
    print(f"Gen {generation}: best fitness = {-fitness:.4f}")
    # Return True to stop early
    return fitness < -0.95  # Stop if NSE > 0.95

result = calibrate(
    ...,
    progress=False,  # Required to use custom callback
    callback=my_callback,
)
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

### "hypsometric_curve is required when n_layers > 1"

**Cause:** You set `n_layers > 1` but did not provide a hypsometric curve.

**Solution:** Provide a 101-point hypsometric curve:

```python
import numpy as np
from gr6j import Catchment

catchment = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,
    hypsometric_curve=np.linspace(200.0, 2000.0, 101),  # 101 points!
    input_elevation=500.0,
)
```

### "input_elevation is required when n_layers > 1"

**Cause:** You set `n_layers > 1` but did not provide the forcing station elevation.

**Solution:** Add `input_elevation` to your Catchment:

```python
catchment = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,
    hypsometric_curve=np.linspace(200.0, 2000.0, 101),
    input_elevation=500.0,  # Add this!
)
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

### "DEM file not found: ..."

**Cause:** The specified DEM file path does not exist.

**Solution:** Check that the file path is correct:

```python
from pathlib import Path

dem_path = "data/basin_dem.tif"
if not Path(dem_path).exists():
    print(f"File not found: {dem_path}")
# Check for typos or use absolute path
```

### "All pixels are NoData or invalid"

**Cause:** The DEM file contains no valid elevation data - all pixels are NoData, NaN, or infinity.

**Solution:** Check your DEM file:

1. Verify the DEM was clipped correctly to the catchment boundary
2. Check the NoData value is set correctly in the raster metadata
3. Open the file in GIS software (QGIS) to inspect visually

```bash
# Check raster info with GDAL
gdalinfo basin_dem.tif
```

### "precip shape ... does not match temp shape ..."

**Cause:** The precipitation and temperature arrays passed to solid precipitation utilities have different shapes.

**Solution:** Ensure both arrays have the same shape:

```python
import numpy as np

# Check shapes before calling
print(f"precip shape: {precip.shape}")
print(f"temp shape: {temp.shape}")

# Ensure they match
assert precip.shape == temp.shape, "Arrays must have the same shape"
```

### "t_snow must be less than t_rain"

**Cause:** The snow temperature threshold is greater than or equal to the rain threshold in solid precipitation utilities.

**Solution:** Ensure `t_snow < t_rain`:

```python
from gr6j.utils import compute_solid_fraction

# Correct: t_snow < t_rain
fraction = compute_solid_fraction(temp, t_snow=-1.0, t_rain=3.0)

# Wrong: t_snow >= t_rain will raise ValueError
# fraction = compute_solid_fraction(temp, t_snow=3.0, t_rain=-1.0)
```

---

## Performance Notes

The GR6J implementation uses [Numba](https://numba.pydata.org/) for just-in-time (JIT) compilation of the core model functions. This provides significant performance improvements over pure Python.

### JIT Compilation Overhead

The first time you run the model, Numba compiles the core functions to machine code. This initial compilation takes a few seconds but only happens once per Python session:

```python
# First run: includes JIT compilation overhead (~1-2 seconds)
output1 = run(params, forcing)

# Subsequent runs: compiled code is reused (very fast)
output2 = run(params, forcing2)  # ~10-50x faster
```

Compiled code is cached to disk (`__pycache__`), so future Python sessions benefit from pre-compiled code after the first run.

### Expected Performance

On typical hardware, the Numba-accelerated implementation achieves:

| Mode | Performance |
|------|-------------|
| GR6J only | ~5-7 million timesteps/second |
| GR6J + single-layer snow | ~3-5 million timesteps/second |
| GR6J + multi-layer snow (5 layers) | ~1-2 million timesteps/second |

This means a 30-year daily simulation (10,950 timesteps) completes in approximately 1-5 milliseconds.

### No User Action Required

The acceleration is transparent - you don't need to change your code to benefit from it. Simply call `run()` as documented and the optimized implementation is used automatically.

### Calibration Performance

During calibration, the model runs thousands of times per optimization. The Numba acceleration provides substantial speedups:

- **Single-objective calibration** (100 generations, 50 individuals): Typically completes in seconds to minutes depending on forcing data length
- **Multi-objective calibration**: Similar performance, with Pareto front evaluation adding minimal overhead

For very large calibration runs, consider using parallel execution:

```python
result = calibrate(
    ...,
    n_workers=-1,  # Use all CPU cores
)
```

---

## Further Reading

- [Model equations and algorithm](MODEL_DEFINITION.md)
- [Snow module technical details](CEMANEIGE.md)
- [Forcing data contract](FORCING_DATA_CONTRACT.md)

## References

- Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T. and Andreassian, V. (2011). A downward structural sensitivity analysis of hydrological models to improve low-flow simulation. *Journal of Hydrology*, 411(1-2), 66-76.

- Valery, A., Andreassian, V., and Perrin, C. (2014). 'As simple as possible but not simpler': What is useful in a temperature-based snow-accounting routine? Part 1 - Comparison of six snow accounting routines on 380 catchments. *Journal of Hydrology*, 517, 1166-1175.
