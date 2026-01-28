# GR6J

🟢 **Active Development** — This repository is part of an ongoing project and actively maintained.

A Python implementation of the **GR6J** (Génie Rural à 6 paramètres Journalier) lumped conceptual rainfall-runoff model for daily streamflow simulation.

## Overview

GR6J is an extension of the widely-used GR4J model, developed by INRAE (France), with an additional **exponential store** to improve low-flow simulation. This implementation includes the optional **CemaNeige** snow module for cold-climate catchments. It operates in simulation mode (concurrent prediction), making it ideal for:

- Ungauged basin prediction
- Climate change impact studies
- Regional hydrological modeling

### Key Features

| Property | Value |
|----------|-------|
| Time step | Daily |
| Spatial resolution | Lumped (catchment-scale) |
| Parameters | 6 GR6J + 2 CemaNeige + 1 Catchment (optional snow) |
| Stores | 3 (Production, Routing, Exponential) |
| Unit hydrographs | 2 (UH1 and UH2) |
| Inputs | Precipitation (P), PET (E), Temperature (T, optional) |
| Output | Streamflow at catchment outlet (Q) |

## Installation

```bash
# Using uv (recommended)
uv add gr6j

# Using pip
pip install gr6j
```

## Usage

```python
import numpy as np
from gr6j import ForcingData, ModelOutput, Parameters, run

# Define model parameters
params = Parameters(
    x1=350.0,   # Production store capacity [mm]
    x2=0.0,     # Intercatchment exchange coefficient [mm/day]
    x3=90.0,    # Routing store capacity [mm]
    x4=1.7,     # Unit hydrograph time constant [days]
    x5=0.0,     # Intercatchment exchange threshold [-]
    x6=5.0,     # Exponential store scale parameter [mm]
)

# Prepare input data
forcing = ForcingData(
    time=np.arange(5, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),  # mm/day
    pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),       # mm/day
)

# Run the model
output = run(params, forcing)

# Access streamflow
print(output.gr6j.streamflow)
print(output.to_dataframe())
```

### Custom Initial State

There are two ways to initialize model state:

**Option 1: Derived from parameters (recommended for fresh runs)**

```python
params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)

# Computes defaults as fractions of capacity:
#   production_store = 0.30 * X1 = 105 mm
#   routing_store    = 0.50 * X3 = 45 mm
#   exponential_store = 0 mm
initial_state = State.initialize(params)
```

**Option 2: Explicit values in mm (useful for warm-starting from a previous run)**

```python
import numpy as np

# Values are direct mm amounts, independent of parameters
custom_state = State(
    production_store=200.0,      # mm
    routing_store=50.0,          # mm
    exponential_store=0.0,       # mm (can be negative)
    uh1_states=np.zeros(20),
    uh2_states=np.zeros(40),
)

output = run(params, forcing, initial_state=custom_state)
```

### Snow Module (CemaNeige)

For cold-climate catchments, enable the CemaNeige snow module to preprocess precipitation through snow accumulation and melt:

```python
import numpy as np
from gr6j import Catchment, CemaNeige, ForcingData, Parameters, run

# Define model parameters with embedded snow module
params = Parameters(
    x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0,
    snow=CemaNeige(ctg=0.97, kf=2.5)  # Snow parameters nested in Parameters
)

# Define catchment properties (required for snow)
catchment = Catchment(mean_annual_solid_precip=150.0)

# Input data must include temperature when snow is enabled
forcing = ForcingData(
    time=np.arange(5, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
    pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
    temp=np.array([-5.0, 0.0, 5.0, -2.0, 8.0]),  # °C
)

# Run with snow module
output = run(params, forcing, catchment=catchment)

# Access snow outputs
print(output.snow.snow_pack)       # Snow water equivalent [mm]
print(output.snow.snow_melt)       # Daily melt [mm/day]
print(output.gr6j.streamflow)      # Total streamflow [mm/day]
```

When snow is enabled, `output.snow` contains 12 fields (11 CemaNeige + 1 precip_raw). Use `output.to_dataframe()` to get all 32 columns.

### Single Timestep Execution

```python
from gr6j import Parameters, State, step
from gr6j.model.unit_hydrographs import compute_uh_ordinates

params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
state = State.initialize(params)
uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

# Execute one timestep
new_state, fluxes = step(state, params, precip=10.0, pet=3.0,
                         uh1_ordinates=uh1_ord, uh2_ordinates=uh2_ord)

print(f"Streamflow: {fluxes['streamflow']:.2f} mm/day")
```

## Model Parameters

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **X1** | Production store capacity | mm | [1, 2500] |
| **X2** | Intercatchment exchange coefficient | mm/day | [-5, 5] |
| **X3** | Routing store capacity | mm | [1, 1000] |
| **X4** | Time constant of unit hydrograph | days | [0.5, 10] |
| **X5** | Intercatchment exchange threshold | - | [-4, 4] |
| **X6** | Exponential store scale parameter | mm | [0.01, 20] |

### CemaNeige Parameters (Snow Module)

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **CTG** | Thermal state weighting coefficient | - | [0, 1] |
| **Kf** | Degree-day melt factor | mm/°C/day | [1, 10] |

Note: `mean_annual_solid_precip` is now specified via the `Catchment` class as it is a static catchment property rather than a calibration parameter.

For detailed CemaNeige equations and algorithm, see [`docs/CEMANEIGE.md`](docs/CEMANEIGE.md).

## Documentation

- Model equations: [`docs/MODEL_DEFINITION.md`](docs/MODEL_DEFINITION.md)
- Snow module: [`docs/CEMANEIGE.md`](docs/CEMANEIGE.md)
- User guide: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- Input validation: [`docs/FORCING_DATA_CONTRACT.md`](docs/FORCING_DATA_CONTRACT.md)

## References

- Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T. and Andréassian, V. (2011). **A downward structural sensitivity analysis of hydrological models to improve low-flow simulation.** *Journal of Hydrology*, 411(1-2), 66-76. [doi:10.1016/j.jhydrol.2011.09.034](https://doi.org/10.1016/j.jhydrol.2011.09.034)

- Valéry, A., Andréassian, V., & Perrin, C. (2014). **'As simple as possible but not simpler': What is useful in a temperature-based snow-accounting routine?** Part 1 – Comparison of six snow accounting routines on 380 catchments. *Journal of Hydrology*, 517, 1166-1175. [doi:10.1016/j.jhydrol.2014.04.059](https://doi.org/10.1016/j.jhydrol.2014.04.059)
