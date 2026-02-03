# HBV-light Model Documentation

## Overview and History

**HBV** (Hydrologiska Byråns Vattenbalansavdelning, Swedish for "Hydrological Bureau Water Balance Section") is a lumped conceptual rainfall-runoff model originally developed at the Swedish Meteorological and Hydrological Institute (SMHI) in the early 1970s by Sten Bergström.

### Development Timeline

- **1972-1976**: Initial development at SMHI by Bergström for operational runoff forecasting in Sweden
- **1976**: First major publication describing the model structure (Bergström, 1976)
- **1990s**: Widespread international adoption and various modifications
- **2005-2012**: Jan Seibert developed **HBV-light**, a user-friendly version with improved calibration tools
- **2012**: Comprehensive description of HBV-light published (Seibert & Vis, 2012)

### Key Characteristics

- **Lumped**: Treats the catchment as a single homogeneous unit
- **Conceptual**: Uses simplified representations of hydrological processes
- **Daily timestep**: Designed for daily simulation (sub-daily versions exist)
- **Parsimonious**: 14 parameters capture dominant rainfall-runoff dynamics
- **Robust**: Successfully applied worldwide across diverse climates

---

## Model Structure

```
                    INPUTS
                      │
         ┌────────────┴────────────┐
         │                         │
    Precipitation (P)        Temperature (T)
         │                         │
         └────────────┬────────────┘
                      ▼
    ┌─────────────────────────────────────┐
    │         SNOW ROUTINE                │
    │  ┌─────────────────────────────┐    │
    │  │      Snow Pack (SP)         │    │
    │  │    ┌───────────────┐        │    │
    │  │    │ Liquid Water  │        │    │
    │  │    │     (LW)      │        │    │
    │  │    └───────────────┘        │    │
    │  └─────────────────────────────┘    │
    │  Parameters: tt, cfmax, sfcf,       │
    │              cwh, cfr               │
    └─────────────────┬───────────────────┘
                      │ Snowmelt + Rain
                      ▼
    ┌─────────────────────────────────────┐
    │          SOIL ROUTINE               │
    │  ┌─────────────────────────────┐    │
    │  │    Soil Moisture (SM)       │    │
    │  │         [0, FC]             │    │
    │  └─────────────────────────────┘    │
    │  Parameters: fc, lp, beta           │
    │                    │                │
    │         ┌──────────┴──────────┐     │
    │         ▼                     ▼     │
    │   Actual ET              Recharge   │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │        RESPONSE ROUTINE             │
    │                                     │
    │  ┌─────────────────────────────┐    │
    │  │   Upper Zone (SUZ)          │────┼──► Q0 (surface, if SUZ > UZL)
    │  │                             │────┼──► Q1 (interflow)
    │  └──────────────┬──────────────┘    │
    │                 │ Percolation       │
    │                 ▼                   │
    │  ┌─────────────────────────────┐    │
    │  │   Lower Zone (SLZ)          │────┼──► Q2 (baseflow)
    │  └─────────────────────────────┘    │
    │  Parameters: k0, k1, k2, perc, uzl  │
    └─────────────────┬───────────────────┘
                      │ Q0 + Q1 + Q2
                      ▼
    ┌─────────────────────────────────────┐
    │        ROUTING ROUTINE              │
    │  ┌─────────────────────────────┐    │
    │  │    Triangular Unit          │    │
    │  │      Hydrograph             │    │
    │  │         /\                  │    │
    │  │        /  \                 │    │
    │  │       /    \                │    │
    │  │      ────────               │    │
    │  │      MAXBAS                 │    │
    │  └─────────────────────────────┘    │
    │  Parameter: maxbas                  │
    └─────────────────┬───────────────────┘
                      │
                      ▼
                   OUTPUT
              Simulated Discharge (Q)
```

---

## Parameters

HBV-light uses 14 parameters organized by routine:

### Snow Routine Parameters

| Parameter | Description | Unit | Default Range | Typical Value |
|-----------|-------------|------|---------------|---------------|
| `tt` | Threshold temperature for rain/snow partitioning | °C | [-2.5, 2.5] | 0.0 |
| `cfmax` | Degree-day factor for snowmelt | mm/°C/d | [0.5, 10.0] | 3.0 |
| `sfcf` | Snowfall correction factor | - | [0.4, 1.4] | 1.0 |
| `cwh` | Water holding capacity of snow (fraction of SP) | - | [0.0, 0.2] | 0.1 |
| `cfr` | Refreezing coefficient | - | [0.0, 0.2] | 0.05 |

### Soil Routine Parameters

| Parameter | Description | Unit | Default Range | Typical Value |
|-----------|-------------|------|---------------|---------------|
| `fc` | Maximum soil moisture storage (field capacity) | mm | [50, 700] | 250 |
| `lp` | Threshold for reduction of ET (fraction of FC) | - | [0.3, 1.0] | 0.9 |
| `beta` | Shape coefficient for recharge function | - | [1.0, 6.0] | 2.0 |

### Response Routine Parameters

| Parameter | Description | Unit | Default Range | Typical Value |
|-----------|-------------|------|---------------|---------------|
| `k0` | Recession coefficient for surface flow | 1/d | [0.05, 0.99] | 0.4 |
| `k1` | Recession coefficient for interflow | 1/d | [0.01, 0.5] | 0.1 |
| `k2` | Recession coefficient for baseflow | 1/d | [0.001, 0.2] | 0.01 |
| `perc` | Maximum percolation rate from SUZ to SLZ | mm/d | [0.0, 6.0] | 1.0 |
| `uzl` | Threshold for surface flow activation | mm | [0.0, 100.0] | 20.0 |

### Routing Parameter

| Parameter | Description | Unit | Default Range | Typical Value |
|-----------|-------------|------|---------------|---------------|
| `maxbas` | Length of triangular weighting function | d | [1.0, 7.0] | 2.5 |

---

## State Variables

| Variable | Description | Unit | Initial Range | Per-Zone |
|----------|-------------|------|---------------|----------|
| `SP` | Snow pack (solid water equivalent) | mm | [0, ~500] | Yes |
| `LW` | Liquid water retained in snow pack | mm | [0, CWH × SP] | Yes |
| `SM` | Soil moisture content | mm | [0, FC] | Yes |
| `SUZ` | Upper zone storage (fast response) | mm | [0, ~100] | No |
| `SLZ` | Lower zone storage (slow response) | mm | [0, ~100] | No |
| `QBuf` | Routing buffer (triangular UH convolution) | mm | Array of length ceil(MAXBAS) | No |

**Note:** When using elevation bands (`n_layers > 1`), SP, LW, and SM are tracked independently for each zone. SUZ, SLZ, and QBuf remain lumped across all zones.

---

## Elevation Bands

HBV-light supports **semi-distributed elevation bands** to account for temperature and precipitation gradients with altitude. This is particularly important in mountainous catchments where snow accumulation and melt timing vary significantly with elevation.

### Key Characteristics

| Property | Value |
|----------|-------|
| Configuration | Optional (default: single zone) |
| Recommended layers | 5 (optimal balance of detail vs complexity) |
| Per-zone processes | Snow routine, Soil routine |
| Lumped processes | Response routine (upper/lower zones), Routing |
| Aggregation method | Area-weighted averaging of recharge |

### Semi-Distributed Mode

In **semi-distributed mode**, the catchment is divided into elevation zones:

1. Each zone runs independent snow and soil routines
2. Temperature and precipitation are extrapolated to each zone's representative elevation
3. Recharge from all zones is aggregated (area-weighted) before entering the response routine
4. The groundwater stores (SUZ, SLZ) and routing remain lumped

### Layer Derivation

Layers are derived from the catchment's **hypsometric curve** - a 101-point cumulative elevation distribution where index `i` contains the elevation at percentile `i%`.

For N layers, each band represents 1/N of the catchment area. The representative elevation for each layer is the midpoint of the elevation range spanned by that layer's percentile bounds.

### Temperature Extrapolation

Temperature decreases linearly with elevation:

```
T_layer = T_input - (GradT / 100) × (Z_layer - Z_input)
```

Where:
- `GradT` = temperature lapse rate [°C/100m] (default: **0.6**)
- `Z_layer` = representative elevation of the layer [m]
- `Z_input` = elevation of forcing data [m]

### Precipitation Extrapolation

Precipitation increases exponentially with elevation (orographic effect):

```
P_layer = P_input × exp(GradP × (Z_layer_eff - Z_input_eff))
```

Where:
- `GradP` = precipitation gradient [m⁻¹] (default: **0.00041**)
- Both elevations are capped at 4000m to prevent unrealistic extrapolation

**Note:** This is the same exponential formula used by CemaNeige, enabling code sharing between models.

### Usage Example

```python
from pydrology import Catchment, ForcingData, get_model
from pydrology.utils import analyze_dem

# Analyze DEM to get hypsometric curve
dem = analyze_dem("data/basin_dem.tif")

# Create catchment with elevation band configuration
catchment = Catchment(
    mean_annual_solid_precip=150.0,
    n_layers=5,                           # 5 equal-area bands
    hypsometric_curve=dem.hypsometric_curve,
    input_elevation=dem.median_elevation,  # Station elevation
    # Optional: custom gradients (defaults used if omitted)
    # temp_gradient=0.6,                  # °C/100m
    # precip_gradient=0.00041,            # m⁻¹
)

# Get model and run
model = get_model("hbv_light")
params = model.Parameters(
    tt=0.0, cfmax=3.0, sfcf=1.0, cwh=0.1, cfr=0.05,
    fc=250.0, lp=0.9, beta=2.0,
    k0=0.4, k1=0.1, k2=0.01, perc=1.0, uzl=20.0,
    maxbas=2.5,
)

result = model.run(params, forcing, catchment=catchment)

# Access aggregated outputs
print(result.streamflow)           # Shape: (n_timesteps,)
print(result.fluxes.snow_pack)     # Aggregated snow pack

# Access per-zone outputs
if result.zone_outputs is not None:
    print(result.zone_outputs.snow_pack)       # Shape: (n_timesteps, 5)
    print(result.zone_outputs.zone_temp)       # Shape: (n_timesteps, 5)
    print(result.zone_outputs.zone_elevations) # Shape: (5,)
```

---

## Mathematical Equations

### Snow Routine

**Precipitation Partitioning:**
```
if T < TT:
    Ps = P × SFCF    (snowfall, corrected)
    Pr = 0           (rainfall)
else:
    Ps = 0
    Pr = P
```

**Snowmelt:**
```
if T > TT and SP > 0:
    Melt = min(CFMAX × (T - TT), SP)
else:
    Melt = 0
```

**Refreezing:**
```
if T < TT and LW > 0:
    Refreeze = min(CFR × CFMAX × (TT - T), LW)
else:
    Refreeze = 0
```

**Snow Pack Update:**
```
SP = SP + Ps - Melt + Refreeze
LW = LW + Melt - Refreeze

# Liquid water release when exceeding holding capacity
if LW > CWH × SP:
    Release = LW - CWH × SP
    LW = CWH × SP
else:
    Release = 0

# Total input to soil routine
I = Pr + Release
```

### Soil Routine

**Recharge to Upper Zone:**
```
Recharge = I × (SM / FC)^BETA
```

**Soil Moisture Update:**
```
SM = SM + I - Recharge
```

**Actual Evapotranspiration:**
```
if SM < LP × FC:
    ETact = PET × (SM / (LP × FC))
else:
    ETact = PET

SM = SM - ETact
```

Where PET is the potential evapotranspiration (input).

### Response Routine

**Surface Flow (Q0):**
```
if SUZ > UZL:
    Q0 = K0 × (SUZ - UZL)
else:
    Q0 = 0
```

**Interflow (Q1):**
```
Q1 = K1 × SUZ
```

**Percolation:**
```
Perc = min(PERC, SUZ)
```

**Upper Zone Update:**
```
SUZ = SUZ + Recharge - Q0 - Q1 - Perc
```

**Baseflow (Q2):**
```
Q2 = K2 × SLZ
```

**Lower Zone Update:**
```
SLZ = SLZ + Perc - Q2
```

**Total Generated Runoff:**
```
Qgen = Q0 + Q1 + Q2
```

### Routing Routine

**Triangular Unit Hydrograph:**

The triangular weighting function has base length `MAXBAS` days:

```
         ▲
    2/MB │    /\
         │   /  \
         │  /    \
         │ /      \
         │/        \
         └──────────► time
         0   MB/2  MB

where MB = MAXBAS
```

**Weights calculation:**
```
for i in range(ceil(MAXBAS)):
    t1 = i
    t2 = i + 1

    # Integrate triangular function over [t1, t2]
    if t2 <= MAXBAS/2:
        # Rising limb
        w[i] = (4/MAXBAS²) × (t2² - t1²) / 2
    elif t1 >= MAXBAS/2:
        # Falling limb
        w[i] = (4/MAXBAS²) × ((MAXBAS-t1)² - (MAXBAS-t2)²) / 2
    else:
        # Spans the peak
        w[i] = ... (split integral)
```

**Routed Discharge:**
```
Q[t] = Σ(w[i] × Qgen[t-i]) for i = 0 to ceil(MAXBAS)-1
```

---

## Usage Examples

### Simulation

```python
from pydrology import ForcingData, get_model

# Load forcing data
forcing = ForcingData(
    precipitation=precip_array,      # mm/d
    temperature=temp_array,          # °C
    pet=pet_array,                   # mm/d
)

# Get model instance
model = get_model("hbv_light")

# Define parameters
params = model.Parameters(
    # Snow routine
    tt=0.0,
    cfmax=3.0,
    sfcf=1.0,
    cwh=0.1,
    cfr=0.05,
    # Soil routine
    fc=250.0,
    lp=0.9,
    beta=2.0,
    # Response routine
    k0=0.4,
    k1=0.1,
    k2=0.01,
    perc=1.0,
    uzl=20.0,
    # Routing
    maxbas=2.5,
)

# Run simulation
result = model.run(params, forcing)

# Access outputs
discharge = result.discharge           # Simulated Q (mm/d)
states = result.states                 # Dictionary of state time series
```

### Single-Timestep Execution

```python
# Initialize state
state = model.State(sp=0.0, lw=0.0, sm=150.0, suz=10.0, slz=30.0)

# Step through each timestep
for t in range(len(forcing)):
    state, q = model.step(
        params=params,
        state=state,
        precip=forcing.precipitation[t],
        temp=forcing.temperature[t],
        pet=forcing.pet[t],
    )
```

### Calibration

```python
from pydrology import calibrate

# Automatic calibration using default parameter bounds
result = calibrate(
    model="hbv_light",
    forcing=forcing,
    observed=observed_discharge,
    objectives=["nse"],
    use_default_bounds=True,
    n_generations=100,
    population_size=50,
)

# Access calibrated parameters
best_params = result.parameters
print(f"NSE: {result.objective_values['nse']:.3f}")

# Custom parameter bounds
custom_bounds = {
    "fc": (100, 500),
    "beta": (1.0, 4.0),
    "k1": (0.05, 0.3),
}

result = calibrate(
    model="hbv_light",
    forcing=forcing,
    observed=observed_discharge,
    objectives=["nse", "logNse"],
    bounds=custom_bounds,  # Override specific parameters
    use_default_bounds=True,  # Use defaults for others
)
```

### Multi-Objective Calibration

```python
result = calibrate(
    model="hbv_light",
    forcing=forcing,
    observed=observed_discharge,
    objectives=["nse", "kge", "pbias"],
    n_generations=200,
)

# Pareto front of solutions
for solution in result.pareto_front:
    print(f"NSE: {solution.nse:.3f}, KGE: {solution.kge:.3f}")
```

---

## Differences from GR6J

| Aspect | HBV-light | GR6J |
|--------|-----------|------|
| **Origin** | Swedish (SMHI, 1970s) | French (INRAE, 1990s) |
| **Snow Module** | Built-in degree-day method | External (CemaNeige coupling) |
| **Total Parameters** | 14 | 6 (+ 6 for CemaNeige = 12) |
| **Storage Reservoirs** | 4 (SP, SM, SUZ, SLZ) | 3 (Production, Routing, Exponential) |
| **Soil Representation** | Single reservoir with FC | Production store with X1 capacity |
| **Recharge Function** | Power function (SM/FC)^β | Percolation based on store level |
| **Fast Response** | Two-layer (SUZ → Q0, Q1) | Single routing store |
| **Baseflow** | Separate lower zone (SLZ → Q2) | Exponential store for low flows |
| **Routing** | Triangular unit hydrograph | S-curve unit hydrograph (UH1, UH2) |
| **Groundwater Exchange** | None | Exchange coefficient (X2) |
| **Inter-catchment Flow** | Not represented | Explicit gain/loss term |
| **ET Reduction** | Linear below LP×FC | Based on production store level |
| **Typical Applications** | Nordic/cold regions | Temperate/Mediterranean |
| **Elevation Bands** | Built-in semi-distributed support | Via CemaNeige coupling |
| **Per-Zone Processing** | Snow + Soil routines | Snow routine only (CemaNeige) |

### When to Choose Each Model

**Choose HBV-light when:**
- Snow processes are important
- You prefer an integrated snow-hydrology model
- Working in Nordic or alpine catchments
- You want explicit control over fast/slow flow separation

**Choose GR6J when:**
- Groundwater exchange is significant
- Low-flow simulation is critical (exponential store)
- You need inter-catchment transfers
- Working in temperate or Mediterranean climates

---

## References

### Primary Sources

1. **Bergström, S.** (1976). Development and application of a conceptual runoff model for Scandinavian catchments. *SMHI Reports RHO No. 7*, Swedish Meteorological and Hydrological Institute, Norrköping.

2. **Bergström, S.** (1995). The HBV model. In: Singh, V.P. (Ed.), *Computer Models of Watershed Hydrology*. Water Resources Publications, Highlands Ranch, CO, pp. 443-476.

3. **Seibert, J., & Vis, M. J. P.** (2012). Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. *Hydrology and Earth System Sciences*, 16(9), 3315-3325. https://doi.org/10.5194/hess-16-3315-2012

### Additional Reading

4. **Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S.** (1997). Development and test of the distributed HBV-96 hydrological model. *Journal of Hydrology*, 201(1-4), 272-288.

5. **Seibert, J.** (1999). Regionalisation of parameters for a conceptual rainfall-runoff model. *Agricultural and Forest Meteorology*, 98-99, 279-293.

6. **Beck, H. E., et al.** (2020). Global-scale evaluation of 22 precipitation datasets using gauge observations and hydrological modeling. *Hydrology and Earth System Sciences*, 24(7), 3585-3612.
