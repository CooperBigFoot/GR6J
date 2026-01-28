# CemaNeige Snow Module Technical Definition

> **Coupled Snow Accumulation and Melt Model**
> A degree-day snow accounting routine for use with GR rainfall-runoff models

---

## Table of Contents

1. [Model Overview](#1-model-overview)
   - [Warm-up Period](#warm-up-period)
2. [Model Parameters](#2-model-parameters)
3. [State Variables](#3-state-variables)
4. [Model Structure Diagram](#4-model-structure-diagram)
5. [Mathematical Equations](#5-mathematical-equations)
   - [5.1 Precipitation Partitioning](#51-precipitation-partitioning)
   - [5.2 Snow Accumulation](#52-snow-accumulation)
   - [5.3 Thermal State Evolution](#53-thermal-state-evolution)
   - [5.4 Potential Melt (Degree-Day)](#54-potential-melt-degree-day)
   - [5.5 Snow Pack Ratio (Gratio)](#55-snow-pack-ratio-gratio)
   - [5.6 Actual Melt](#56-actual-melt)
   - [5.7 Output to Hydrological Model](#57-output-to-hydrological-model)
6. [Hysteresis Mode (Optional)](#6-hysteresis-mode-optional)
7. [Complete Algorithm](#7-complete-algorithm)
8. [Numerical Constants](#8-numerical-constants)
9. [Coupling with GR6J](#9-coupling-with-gr6j)
10. [Elevation Bands](#10-elevation-bands)
11. [Calibration Strategy](#11-calibration-strategy)
12. [Known Limitations](#12-known-limitations)
13. [Model Outputs](#13-model-outputs)
14. [References](#14-references)
15. [Appendix A: Fortran Variable Mapping](#appendix-a-fortran-variable-mapping)
16. [Appendix B: Symbol Reference](#appendix-b-symbol-reference)

---

## 1. Model Overview

**CemaNeige** (Coupled Snow Accumulation and Melt Model) is a temperature-based snow accounting routine developed by INRAE (France). It is designed to be coupled with GR rainfall-runoff models (GR4J, GR5J, GR6J) to handle snow processes in mountainous or cold-climate catchments.

### Key Characteristics

| Property | Value |
|----------|-------|
| Time step | Daily |
| Spatial resolution | Lumped or semi-distributed (elevation bands) |
| Parameters | 2 (standard) or 4 (with hysteresis) |
| State variables | 4 |
| Inputs | Precipitation (P), Temperature (T), Solid Fraction |
| Output | Liquid water available for runoff (PliqAndMelt) |

### Model Philosophy

CemaNeige uses a **degree-day approach** for snow melt, which assumes melt is proportional to air temperature above a threshold (0°C). This approach is:

- Computationally efficient
- Requires only temperature data (no radiation)
- Validated across 380+ catchments globally

### Warm-up Period

Like GR6J, CemaNeige requires a warm-up period to initialize snow pack state:

| Aspect | Recommendation |
|--------|----------------|
| Duration | **365 days** (1 year) minimum |
| Purpose | Allow snow pack to reach realistic levels |
| Cold climates | Use 2+ years for deep snow packs |

**Why warm-up is needed:**

- Initial snow pack (G) is typically set to 0 mm
- Thermal state (eTG) starts at 0°C
- These defaults rarely match actual catchment conditions at simulation start

---

## 2. Model Parameters

### Standard Mode (2 Parameters)

| Parameter | Symbol | Description | Unit | Typical Range | Calibration Bound |
|-----------|--------|-------------|------|---------------|-------------------|
| **CTG** | X1 | Thermal state weighting coefficient | - | [0, 1] | [0, 1] |
| **Kf** | X2 | Degree-day melt factor | mm/°C/day | [1, 10] | [0, 200]* |

*Note: The calibration upper bound for Kf is 200 mm/°C/day with log-space transformation, though typical calibrated values are 2-5 mm/°C/day.

### Hysteresis Mode (4 Parameters)

| Parameter | Symbol | Description | Unit | Typical Range | Calibration Bound |
|-----------|--------|-------------|------|---------------|-------------------|
| **CTG** | X1 | Thermal state weighting coefficient | - | [0, 1] | [0, 1] |
| **Kf** | X2 | Degree-day melt factor | mm/°C/day | [1, 10] | [0, 200]* |
| **Gacc** | X3 | Accumulation threshold parameter | mm | [50, 250] | [0, 100] |
| **prct** | X4 | Threshold percentage of annual solid precip | - | [0.5, 1.0] | [0.5, 1.5] |

### Typical Calibrated Values

Based on airGR documentation examples across multiple catchments:

| Parameter | Typical Value | Range Observed |
|-----------|---------------|----------------|
| **CTG** | 0.97 | 0.96 - 0.98 |
| **Kf** | 2.5 mm/°C/day | 2.2 - 2.8 |
| **Gacc** | 100 mm | - |
| **prct** | 0.4 | - |

### Physical Interpretation

- **CTG (Thermal state coefficient)**: Controls the thermal inertia of the snow pack. Higher values (close to 1) mean the snow pack temperature changes slowly; lower values mean rapid adjustment to air temperature.

- **Kf (Degree-day factor)**: Controls how much snow melts per degree above 0°C. Typical values range from 1-5 mm/°C/day depending on snow properties and local conditions.

- **Gacc (Accumulation parameter)**: Controls how quickly the snow cover fraction (Gratio) increases during accumulation. Only used in hysteresis mode.

- **prct (Percentage parameter)**: Defines the initial melt threshold as a fraction of mean annual solid precipitation. Only used in hysteresis mode.

**Note on mean_annual_solid_precip:** In earlier versions, `mean_annual_solid_precip` was
a parameter of the `CemaNeige` class. It has now been moved to the `Catchment` class since
it is a static catchment property rather than a calibration parameter.

```python
# Old API (deprecated)
snow = CemaNeige(ctg=0.97, kf=2.5, mean_annual_solid_precip=150.0)

# New API
params = Parameters(..., snow=CemaNeige(ctg=0.97, kf=2.5))
catchment = Catchment(mean_annual_solid_precip=150.0)
output = run(params, forcing, catchment=catchment)
```

---

## 3. State Variables

The model maintains **4 state variables**:

| State | Symbol | Description | Unit | Initialization |
|-------|--------|-------------|------|----------------|
| Snow pack | G | Snow water equivalent | mm | 0 mm |
| Thermal state | eTG | Weighted snow temperature | °C | 0°C |
| Melt threshold | Gthreshold | Snow pack threshold for melt | mm | 0.9 × MeanAnSolidPrecip |
| Local maximum | Glocalmax | Hysteresis memory variable | mm | Gthreshold |

**Key constraints:**

- G ≥ 0 (snow pack cannot be negative)
- eTG ≤ 0 (snow temperature cannot exceed 0°C)
- Gratio ∈ [0, 1] (snow cover fraction bounded)

---

## 4. Model Structure Diagram

```
                         INPUTS
                           │
              ┌────────────┼────────────┐
              │            │            │
        Precipitation  Temperature  Solid Fraction
              │            │            │
              └────────────┴────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  PRECIPITATION SPLIT   │
              │                        │
              │  Pliq = (1-f) × P      │
              │  Psol = f × P          │
              └────────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         Liquid (Pliq)            Solid (Psol)
              │                         │
              │                         ▼
              │            ┌────────────────────────┐
              │            │      SNOW PACK         │
              │            │         (G)            │
              │            │                        │
              │            │   G = G + Psol         │
              │            └────────────┬───────────┘
              │                         │
              │            ┌────────────┴────────────┐
              │            │    THERMAL STATE        │
              │            │        (eTG)            │
              │            │                         │
              │            │ eTG = CTG×eTG +         │
              │            │       (1-CTG)×T         │
              │            │ eTG = min(eTG, 0)       │
              │            └────────────┬────────────┘
              │                         │
              │                         ▼
              │            ┌────────────────────────┐
              │            │    DEGREE-DAY MELT     │
              │            │                        │
              │            │ If eTG=0 and T>0:      │
              │            │   PotMelt = Kf × T     │
              │            │ Else:                  │
              │            │   PotMelt = 0          │
              │            └────────────┬───────────┘
              │                         │
              │                         ▼
              │            ┌────────────────────────┐
              │            │   MELT MODULATION      │
              │            │      (Gratio)          │
              │            │                        │
              │            │ Gratio = G/Gthreshold  │
              │            │ Melt = f(Gratio) ×     │
              │            │        PotMelt         │
              │            └────────────┬───────────┘
              │                         │
              │                    Actual Melt
              │                         │
              │                         ▼
              │            ┌────────────────────────┐
              │            │   SNOW PACK UPDATE     │
              │            │                        │
              │            │     G = G - Melt       │
              │            └────────────┬───────────┘
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    COMBINE OUTPUTS     │
              │                        │
              │ PliqAndMelt = Pliq +   │
              │              Melt      │
              └────────────┬───────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │   TO GR MODEL   │
                 │  (as Precip)    │
                 └─────────────────┘
```

---

## 5. Mathematical Equations

### 5.1 Precipitation Partitioning

Total precipitation is split into liquid and solid fractions:

**Liquid precipitation:**
$$P_{liq} = (1 - f_{solid}) \times P$$

**Solid precipitation (snow):**
$$P_{sol} = f_{solid} \times P$$

Where $f_{solid}$ is the fraction of solid precipitation, typically computed from temperature using the USACE formula:

$$f_{solid} = \begin{cases}
1 & \text{if } T \leq -1°C \\
\frac{3 - T}{4} & \text{if } -1°C < T < 3°C \\
0 & \text{if } T \geq 3°C
\end{cases}$$

---

### 5.2 Snow Accumulation

The snow pack accumulates solid precipitation:

$$G_{init} = G$$
$$G = G + P_{sol}$$

Where:
- $G_{init}$ = snow pack at start of time step (stored for hysteresis calculations)
- $G$ = updated snow pack after accumulation

---

### 5.3 Thermal State Evolution

The thermal state uses exponential smoothing to represent snow pack thermal inertia:

$$eTG = CTG \times eTG + (1 - CTG) \times T$$

**Constraint (snow cannot be warmer than 0°C):**
$$eTG = \min(eTG, 0)$$

**Physical interpretation:**
- When CTG → 1: Snow pack temperature changes very slowly (high thermal inertia)
- When CTG → 0: Snow pack temperature tracks air temperature closely
- The constraint ensures snow temperature never exceeds melting point

---

### 5.4 Potential Melt (Degree-Day)

Melt only occurs when the snow pack is thermally equilibrated (eTG = 0) and air temperature exceeds the melt threshold:

$$PotMelt = \begin{cases}
K_f \times (T - T_{melt}) & \text{if } eTG = 0 \text{ and } T > T_{melt} \\
0 & \text{otherwise}
\end{cases}$$

Where $T_{melt} = 0°C$ (fixed constant).

**Constraint (cannot melt more than available):**
$$PotMelt = \min(PotMelt, G)$$

---

### 5.5 Snow Pack Ratio (Gratio)

The snow pack ratio controls what fraction of potential melt becomes actual melt.

> **Important:** In standard mode, Gratio is calculated **twice per timestep**:
> 1. **Before melt** (using G after accumulation) - used to compute actual melt
> 2. **After melt** (using G after melt subtraction) - stored in output
>
> This means the output Gratio reflects the snow pack state AFTER melt has occurred.

#### Standard Mode

$$Gratio = \begin{cases}
\frac{G}{G_{threshold}} & \text{if } G < G_{threshold} \\
1 & \text{if } G \geq G_{threshold}
\end{cases}$$

Where:
$$G_{threshold} = 0.9 \times MeanAnSolidPrecip$$

#### Hysteresis Mode

See [Section 6](#6-hysteresis-mode-optional) for the more complex hysteresis formulation.

---

### 5.6 Actual Melt

Actual melt is modulated by the snow pack ratio:

$$Melt = \left((1 - MinSpeed) \times Gratio + MinSpeed\right) \times PotMelt$$

Where $MinSpeed = 0.1$ (10% minimum melt rate).

**Physical interpretation:**
- When Gratio = 1 (full snow cover): Melt = 100% of PotMelt
- When Gratio = 0 (nearly depleted): Melt = 10% of PotMelt (minimum)
- The MinSpeed ensures melt never completely stops when snow exists

**Snow pack update:**
$$G = G - Melt$$

---

### 5.7 Output to Hydrological Model

The total liquid water output combines liquid precipitation and snow melt:

$$PliqAndMelt = P_{liq} + Melt$$

This value is passed to the GR model as the **effective precipitation input**.

---

## 6. Hysteresis Mode (Optional)

The hysteresis extension (Riboust et al., 2019) captures the asymmetry between snow accumulation and depletion phases. Snow accumulation is typically homogeneous (covers entire catchment), while melt is heterogeneous (patchy).

### Hysteresis Equations

#### During Potential Melt Conditions

When $PotMelt > 0$:

$$\text{If } G < G_{localmax} \text{ and } Gratio = 1: \quad G_{localmax} = G$$

$$Gratio = \min\left(\frac{G}{G_{localmax}}, 1\right)$$

#### After Melt - Update Gratio

**Accumulation phase** ($dG = G - G_{init} > 0$):

$$Gratio = \min\left(Gratio + \frac{P_{sol} - Melt}{G_{acc}}, 1\right)$$

$$\text{If } Gratio = 1: \quad G_{localmax} = G_{threshold}$$

**Depletion phase** ($dG \leq 0$):

$$Gratio = \min\left(\frac{G}{G_{localmax}}, 1\right)$$

### Initial Threshold (Hysteresis Mode)

$$G_{threshold} = prct \times MeanAnSolidPrecip$$

Where $prct$ is parameter X4.

---

## 7. Complete Algorithm

```
INPUT: P (precipitation), T (temperature), f_solid (solid fraction)
INPUT: MeanAnSolidPrecip (annual mean solid precipitation)
STATE: G (snow pack), eTG (thermal state), Gthreshold, Glocalmax

FOR each time step:

  1. PRECIPITATION PARTITIONING
     Pliq = (1 - f_solid) * P
     Psol = f_solid * P

  2. SNOW ACCUMULATION
     Ginit = G
     G = G + Psol

  3. THERMAL STATE UPDATE
     eTG = CTG * eTG + (1 - CTG) * T
     IF eTG > 0:
       eTG = 0
     ENDIF

  4. POTENTIAL MELT (Degree-Day)
     IF eTG == 0 AND T > 0:
       PotMelt = Kf * T
       IF PotMelt > G:
         PotMelt = G
       ENDIF
     ELSE:
       PotMelt = 0
     ENDIF

  5. GRATIO CALCULATION (before melt)
     IF IsHysteresis:
       [See Section 6 for hysteresis logic]
     ELSE:
       IF G < Gthreshold:
         Gratio = G / Gthreshold
       ELSE:
         Gratio = 1.0
       ENDIF
     ENDIF

  6. ACTUAL MELT
     Melt = ((1 - MinSpeed) * Gratio + MinSpeed) * PotMelt
     G = G - Melt

  7. POST-MELT GRATIO UPDATE
     IF IsHysteresis:
       [See Section 6 for hysteresis logic]
     ELSE:
       // Standard mode: recalculate Gratio after melt
       IF G < Gthreshold:
         Gratio = G / Gthreshold
       ELSE:
         Gratio = 1.0
       ENDIF
     ENDIF

  8. OUTPUT
     PliqAndMelt = Pliq + Melt

OUTPUT: PliqAndMelt (to GR model as precipitation)
        Store final states: G, eTG, Gthreshold, Glocalmax
```

---

## 8. Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Tmelt | 0.0 | Melt temperature threshold [°C] |
| MinSpeed | 0.1 | Minimum melt fraction [-] |
| Default Gthreshold factor | 0.9 | Fraction of MeanAnSolidPrecip for standard mode |

### Solid Fraction Thresholds (USACE Formula)

| Threshold | Value | Description |
|-----------|-------|-------------|
| T_snow | -1°C | Below this, all precipitation is snow |
| T_rain | 3°C | Above this, all precipitation is rain |

---

## 9. Coupling with GR6J

CemaNeige operates as a **preprocessing layer** before the GR6J model:

```
┌─────────────────────────────────────────────────────────────┐
│                    COUPLED MODEL CHAIN                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Raw Inputs: [Precip, Temperature, PET]                     │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │      CEMA-NEIGE       │                       │
│              │   (Snow Module)       │                       │
│              │                       │                       │
│              │  • Snow/rain split    │                       │
│              │  • Accumulation       │                       │
│              │  • Melt calculation   │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│              Output: PliqAndMelt (effective precipitation)   │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │        GR6J          │                       │
│              │  (Hydrological Model) │                       │
│              │                       │                       │
│              │  • Production store   │                       │
│              │  • Routing store      │                       │
│              │  • Exponential store  │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│                          ▼                                   │
│              Output: Streamflow [mm/day]                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Python API Example

```python
import numpy as np
from gr6j import Catchment, CemaNeige, ForcingData, Parameters, run

# Model parameters with embedded snow module
params = Parameters(
    x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0,
    snow=CemaNeige(ctg=0.97, kf=2.5)
)

# Catchment properties
catchment = Catchment(mean_annual_solid_precip=150.0)

# Forcing data with temperature
forcing = ForcingData(
    time=np.arange(365, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=precip_array,
    pet=pet_array,
    temp=temp_array,
)

# Run coupled model
output = run(params, forcing, catchment=catchment)

# Access outputs
print(output.gr6j.streamflow)    # GR6J streamflow
print(output.snow.snow_pack)     # Snow pack water equivalent
print(output.snow.snow_melt)     # Daily melt
```

### Parameter Count for Combined Models

| Model | GR Params | CemaNeige (Standard) | CemaNeige (Hysteresis) | Total |
|-------|-----------|---------------------|------------------------|-------|
| CemaNeigeGR4J | 4 | 2 | 4 | 6 or 8 |
| CemaNeigeGR5J | 5 | 2 | 4 | 7 or 9 |
| CemaNeigeGR6J | 6 | 2 | 4 | 8 or 10 |

### State Variables for Combined Models

| Component | States | Description |
|-----------|--------|-------------|
| GR6J | 3 + 60 | S, R, Exp + UH1[20] + UH2[40] |
| CemaNeige | 4 × NLayers | G, eTG, Gthreshold, Glocalmax per layer |

---

## 10. Elevation Bands

CemaNeige is typically run on **multiple elevation bands** to account for temperature and precipitation gradients with altitude.

### Recommended Configuration

| Setting | Value |
|---------|-------|
| Number of layers | 5 (optimal) |
| Hypsometric curve | 101-point elevation distribution |

### Temperature Lapse Rate

Temperature is extrapolated to each elevation band using daily gradients:

$$T_{layer} = T_{input} + (Z_{input} - Z_{layer}) \times \frac{GradT}{100}$$

Where:
- $Z_{input}$ = elevation of input data [m]
- $Z_{layer}$ = representative elevation of layer [m]
- $GradT$ = temperature gradient [°C/100m] (varies by day of year)

### Precipitation Gradient

Precipitation is extrapolated using an exponential gradient (Valéry, 2010):

$$P_{layer} = P_{input} \times \exp(GradP \times (Z_{layer} - Z_{input}))$$

Where $GradP = 0.00041$ m⁻¹ (capped at 4000m elevation).

### Layer Aggregation

The final output is the area-weighted average across all layers:

$$PliqAndMelt_{catchment} = \frac{1}{N_{layers}} \sum_{i=1}^{N_{layers}} PliqAndMelt_i$$

---

## 11. Calibration Strategy

### Parameter Transformation

The airGR package uses transformed parameter space during calibration:

| Parameter | Transformation | Formula (T → Real) |
|-----------|---------------|-------------------|
| **X1 (CTG)** | Linear | `(T + 9.99) / 19.98` |
| **X2 (Kf)** | Exponential | `exp(T) / 200` |
| **X3 (Gacc)** | Linear | `(T × 5) + 50` |
| **X4 (prct)** | Linear | `(T / 19.98) + 0.5` |

The exponential transformation for Kf ensures the parameter remains positive and allows efficient exploration of values spanning orders of magnitude.

### Hysteresis Mode Calibration

For hysteresis mode (Riboust et al., 2019), the recommended calibration strategy uses a **composite objective function**:

| Component | Weight | Metric |
|-----------|--------|--------|
| Discharge | 75% | KGE' |
| SCA Band 1 | 5% | KGE' |
| SCA Band 2 | 5% | KGE' |
| SCA Band 3 | 5% | KGE' |
| SCA Band 4 | 5% | KGE' |
| SCA Band 5 | 5% | KGE' |

This requires:
- MODIS or similar satellite Snow Cover Area (SCA) data
- 5 elevation bands (optimal configuration)
- Filtering of days with >40% cloud coverage

### MeanAnSolidPrecip Calculation

The annual mean solid precipitation must be computed for each elevation band:

```python
# For each layer, compute mean daily solid precip over calibration period
mean_daily_solid = mean(frac_solid[cal_period] * precip[cal_period])
# Convert to annual
mean_annual_solid = mean_daily_solid * 365.25
```

---

## 12. Known Limitations

### Flow Prediction Biases

- **Low flows**: Tend to be overestimated by CemaNeige-GR models
- **Peak flows**: Also tend to be overestimated in some catchments

### Model Assumptions

- **Degree-day approach**: Cannot represent energy-balance processes; no radiation input
- **Uniform snow properties**: Assumes homogeneous snow within elevation bands
- **Constant Kf**: The degree-day factor is constant, though it should ideally vary seasonally and spatially
- **Daily timestep only**: Model is designed and validated for daily timestep; not suitable for sub-daily or monthly applications

### Calibration Complexity

- Adding CemaNeige to GR models **significantly affects the calibration process**
- More complex than calibrating standalone GR models
- Multi-objective calibration recommended when using hysteresis mode

### Input Data Sensitivity

- Performance is sensitive to quality of gridded meteorological products
- Temperature and precipitation gradients are critical for elevation band implementation
- Regional variability in performance based on input data sources

---

## 13. Model Outputs

CemaNeige produces the following outputs at each time step:

| Index | Variable | Description | Unit |
|-------|----------|-------------|------|
| 1 | Pliq | Liquid precipitation | mm/day |
| 2 | Psol | Solid precipitation | mm/day |
| 3 | G | Snow pack (SWE) | mm |
| 4 | eTG | Thermal state | °C |
| 5 | Gratio | Snow cover fraction | - |
| 6 | PotMelt | Potential melt | mm/day |
| 7 | Melt | Actual melt | mm/day |
| 8 | PliqAndMelt | **Water to GR model** | mm/day |
| 9 | Temp | Air temperature | °C |
| 10 | Gthreshold | Melt threshold | mm |
| 11 | Glocalmax | Local maximum (hysteresis) | mm |

---

## 14. References

### Primary References

> Valéry, A., Andréassian, V., & Perrin, C. (2014).
> **'As simple as possible but not simpler': What is useful in a temperature-based snow-accounting routine?**
> Part 1 – Comparison of six snow accounting routines on 380 catchments.
> *Journal of Hydrology*, 517, 1166-1175.
> doi: [10.1016/j.jhydrol.2014.04.059](https://doi.org/10.1016/j.jhydrol.2014.04.059)

> Valéry, A., Andréassian, V., & Perrin, C. (2014).
> **'As simple as possible but not simpler': What is useful in a temperature-based snow-accounting routine?**
> Part 2 – Sensitivity analysis of the Cemaneige snow accounting routine on 380 catchments.
> *Journal of Hydrology*, 517, 1176-1187.
> doi: [10.1016/j.jhydrol.2014.04.058](https://doi.org/10.1016/j.jhydrol.2014.04.058)

### Hysteresis Extension

> Riboust, P., Thirel, G., Le Moine, N., & Ribstein, P. (2019).
> **Revisiting a simple degree-day model for integrating satellite data: Implementation of SWE-SCA hystereses.**
> *Journal of Hydrology and Hydromechanics*, 67(1), 70-81.
> doi: [10.2478/johh-2018-0004](https://doi.org/10.2478/johh-2018-0004)

### Software Implementations

- **airGR** (R): Official implementation by INRAE
  https://gitlab.irstea.fr/HYCAR-Hydro/airgr

- **RRMPG** (Python): Rainfall-Runoff Modelling Playground
  https://rrmpg.readthedocs.io/

- **cponc8/CemaNeige** (Python): Community implementation
  https://github.com/cponc8/CemaNeige

- **hckaraman/CemaNeige-Snow-Model** (Python): Alternative implementation
  https://github.com/hckaraman/CemaNeige-Snow-Model

---

## Appendix A: Fortran Variable Mapping

For users referencing the official airGR Fortran implementation (`frun_CEMANEIGE.f90`):

### State Variables

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| G | St(1) / G | Snow pack water equivalent |
| eTG | St(2) / eTG | Thermal state |
| Gthreshold | St(3) / Gthreshold | Melt threshold |
| Glocalmax | St(4) / Glocalmax | Local maximum (hysteresis) |

### Model Parameters

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| CTG | Param(1) | Thermal weighting coefficient |
| Kf | Param(2) | Degree-day melt factor |
| Gacc | Param(3) | Accumulation parameter (hysteresis) |
| prct | Param(4) | Threshold percentage (hysteresis) |

### Key Intermediate Variables

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| Pliq | Pliq | Liquid precipitation |
| Psol | Psol | Solid precipitation |
| PotMelt | PotMelt | Potential melt |
| Melt | Melt | Actual melt |
| Gratio | Gratio | Snow pack ratio |
| PliqAndMelt | PliqAndMelt | Output to GR model |

---

## Appendix B: Symbol Reference

### Primary Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| P | Total precipitation | mm/day |
| T | Air temperature | °C |
| G | Snow pack (SWE) | mm |
| eTG | Thermal state | °C |

### Model Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| CTG | Thermal weighting coefficient | - |
| Kf | Degree-day melt factor | mm/°C/day |
| Gacc | Accumulation parameter | mm |
| prct | Threshold percentage | - |

### Intermediate Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| Pliq | Liquid precipitation | mm/day |
| Psol | Solid precipitation | mm/day |
| f_solid | Solid precipitation fraction | - |
| PotMelt | Potential melt | mm/day |
| Melt | Actual melt | mm/day |
| Gratio | Snow pack ratio | - |
| Gthreshold | Melt threshold | mm |
| Glocalmax | Local maximum | mm |
| PliqAndMelt | Water output to GR model | mm/day |

### Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| Tmelt | 0°C | Melt temperature threshold |
| MinSpeed | 0.1 | Minimum melt fraction |

---

*Document generated from analysis of airGR Fortran implementation (frun_CEMANEIGE.f90).*
