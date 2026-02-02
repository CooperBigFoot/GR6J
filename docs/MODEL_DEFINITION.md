# GR6J Model Technical Definition

> **Génie Rural à 6 paramètres Journalier**
> A lumped conceptual rainfall-runoff model for daily streamflow simulation

This document describes the GR6J model as implemented in `pydrology.models.gr6j`.

---

## Table of Contents

1. [Model Overview](#1-model-overview)
   - [Warm-up Period](#warm-up-period)
2. [Model Parameters](#2-model-parameters)
3. [State Variables](#3-state-variables)
4. [Model Structure Diagram](#4-model-structure-diagram)
5. [Mathematical Equations](#5-mathematical-equations)
   - [5.1 Production Store](#51-production-store)
   - [5.2 Percolation](#52-percolation)
   - [5.3 Unit Hydrographs](#53-unit-hydrographs)
   - [5.4 Groundwater Exchange](#54-groundwater-exchange)
   - [5.5 Routing Store](#55-routing-store)
   - [5.6 Exponential Store](#56-exponential-store)
   - [5.7 Direct Branch](#57-direct-branch)
   - [5.8 Total Streamflow](#58-total-streamflow)
6. [Complete Algorithm](#6-complete-algorithm)
7. [Numerical Constants](#7-numerical-constants)
8. [Model Outputs](#8-model-outputs)
9. [Key Differences: GR6J vs GR4J](#9-key-differences-gr6j-vs-gr4j)
10. [References](#10-references)
11. [Appendix A: Fortran Variable Mapping](#appendix-a-fortran-variable-mapping)
12. [Appendix B: Symbol Reference](#appendix-b-symbol-reference)

---

## 1. Model Overview

**GR6J** (Génie Rural à 6 paramètres Journalier) is a lumped, conceptual, daily rainfall-runoff model developed by INRAE (formerly IRSTEA/Cemagref) in France. It is an extension of the widely-used GR4J model, with an additional **exponential store** to improve low-flow simulation.

### Key Characteristics

| Property | Value |
|----------|-------|
| Time step | Daily |
| Spatial resolution | Lumped (catchment-scale) |
| Parameters | 6 calibrated parameters |
| Stores | 3 (Production, Routing, Exponential) |
| Unit hydrographs | 2 (UH1 and UH2) |
| Inputs | Precipitation (P), Potential Evapotranspiration (E) |
| Output | Streamflow at catchment outlet (Q) |

### Model Philosophy

GR6J operates in **simulation mode** (concurrent prediction), meaning it does not use past observed streamflow as input. This makes it ideal for:
- Ungauged basin prediction
- Climate change impact studies
- Regional hydrological modeling

### Warm-up Period

The model requires a **warm-up period** to initialize internal states before producing reliable outputs. During this period:

| Aspect | Recommendation |
|--------|----------------|
| Duration | **365 days** (1 year) minimum |
| Purpose | Allow stores to reach dynamic equilibrium |
| Outputs | Should be discarded from analysis |

**Why warm-up is needed:**
- Initial store levels (S, R, Exp) are typically set to default fractions of capacity
- These default values rarely match actual catchment conditions
- The warm-up allows the model states to "spin up" to realistic values based on observed forcing data

**Best practices:**
1. Use at least 1 year of data before the period of interest
2. For highly seasonal climates, use 2+ years of warm-up
3. Initialize states from a previous run if available (reduces warm-up needs)

---

## 2. Model Parameters

GR6J uses **6 calibrated parameters** that control different aspects of the rainfall-runoff transformation:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **X1** | Production store capacity | mm | [1, 2500] |
| **X2** | Intercatchment exchange coefficient | mm/day | [-5, 5] |
| **X3** | Routing store capacity | mm | [1, 1000] |
| **X4** | Time constant of unit hydrograph | days | [0.5, 10] |
| **X5** | Intercatchment exchange threshold | - | [-4, 4] |
| **X6** | Exponential store scale parameter | mm | [0.01, 20] |

### Physical Interpretation

- **X1 (Production store capacity)**: Controls the maximum soil moisture storage. Larger values mean more water can be retained in the catchment before generating runoff.

- **X2 (Exchange coefficient)**: Controls groundwater exchange with neighboring catchments or deep aquifers. Positive values indicate water import; negative values indicate water export.

- **X3 (Routing store capacity)**: Controls the size of the groundwater reservoir that generates baseflow. Affects recession characteristics.

- **X4 (Unit hydrograph time constant)**: Controls how quickly surface runoff reaches the outlet. Smaller values = faster response.

- **X5 (Exchange threshold)**: Dimensionless threshold controlling when groundwater exchange reverses direction based on routing store level.

- **X6 (Exponential store scale parameter)**: Controls the scale of the exponential store outflow via a softplus function (GR6J-specific). Larger values increase baseflow contribution; smaller values reduce it.

---

## 3. State Variables

The model maintains **3 stores** with evolving water levels:

| Store | Symbol | Description | Initialization |
|-------|--------|-------------|----------------|
| Production store | S | Soil moisture reservoir | 30% of X1 |
| Routing store | R | Groundwater reservoir | 50% of X3 |
| Exponential store | Exp | Slow drainage reservoir (can be negative) | 0 mm |

Additionally, the model maintains **unit hydrograph states**:
- **UH1 states**: 20-element array for slow branch routing
- **UH2 states**: 40-element array for fast branch routing

---

## 4. Model Structure Diagram

```
                         INPUTS
                           │
              ┌────────────┴────────────┐
              │                         │
        Precipitation (P)      Potential ET (E)
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    PRODUCTION STORE    │
              │      (capacity X1)     │
              │                        │
              │  • Net P/E calculation │
              │  • Evapotranspiration  │
              │  • Soil storage        │
              └────────────┬───────────┘
                           │
                     Percolation
                           │
                           ▼
                  Effective Rainfall (PR)
                           │
              ┌────────────┴────────────┐
              │                         │
          90% (b)                    10% (1-b)
              │                         │
              ▼                         ▼
       ┌──────────────┐         ┌──────────────┐
       │     UH1      │         │     UH2      │
       │  (20 days)   │         │  (40 days)   │
       └──────┬───────┘         └──────┬───────┘
              │                        │
    ┌─────────┴─────────┐              │
    │                   │              │
  60% (1-c)          40% (c)           │
    │                   │              │
    ▼                   ▼              │
┌─────────┐      ┌────────────┐        │
│ ROUTING │◄─────│ EXCHANGE   │────────┼────► Exchange (F)
│  STORE  │      │  F = X2 *  │        │
│  (X3)   │      │(R/X3 - X5) │        │
└────┬────┘      └────────────┘        │
     │                   │             │
     │                   ▼             │
     │           ┌────────────┐        │
     │           │EXPONENTIAL │        │
     │           │   STORE    │        │
     │           │   (X6)     │        │
     │           └─────┬──────┘        │
     │                 │               │
     ▼                 ▼               ▼
    QR              QRExp             QD
     │                 │               │
     └────────┬────────┴───────┬───────┘
              │                │
              ▼                │
        ┌───────────┐          │
        │   TOTAL   │◄─────────┘
        │ STREAMFLOW│
        │  Q = QR + │
        │  QRExp+QD │
        └───────────┘
              │
              ▼
           OUTPUT
```

---

## 5. Mathematical Equations

### 5.1 Production Store

The production store partitions precipitation into evapotranspiration, soil storage, and effective rainfall.

#### Case 1: Evapotranspiration Dominant (P < E)

When precipitation is less than potential evapotranspiration:

**Net evapotranspiration:**
$$E_n = E - P$$

**Scaled evapotranspiration (with numerical safeguard):**
$$W_S = \min\left(\frac{E_n}{X_1}, 13\right)$$

**Hyperbolic tangent weighting:**
$$\tanh(W_S) = \frac{e^{2W_S} - 1}{e^{2W_S} + 1}$$

**Evaporation from production store:**
$$E_S = S \cdot \frac{(2 - S/X_1) \cdot \tanh(W_S)}{1 + (1 - S/X_1) \cdot \tanh(W_S)}$$

**Actual evapotranspiration:**
$$E_a = E_S + P$$

**Production store update:**
$$S_{new} = S - E_S$$

**Effective rainfall:**
$$P_n = 0, \quad P_R = 0$$

#### Case 2: Rainfall Dominant (P ≥ E)

When precipitation exceeds potential evapotranspiration:

**Net rainfall:**
$$P_n = P - E$$

**Actual evapotranspiration:**
$$E_a = E$$

**Scaled precipitation (with numerical safeguard):**
$$W_S = \min\left(\frac{P_n}{X_1}, 13\right)$$

**Storage infiltration:**
$$P_S = X_1 \cdot \frac{(1 - (S/X_1)^2) \cdot \tanh(W_S)}{1 + (S/X_1) \cdot \tanh(W_S)}$$

**Rainfall excess:**
$$P_R = P_n - P_S$$

**Production store update:**
$$S_{new} = S + P_S$$

---

### 5.2 Percolation

Water percolates from the production store to deeper layers:

$$Perc = S \cdot \left(1 - \left(1 + \left(\frac{S}{(9/4) \cdot X_1}\right)^4\right)^{-0.25}\right)$$

**Simplified form:**
$$Perc = S \cdot \left(1 - \frac{1}{\sqrt[4]{1 + \left(\frac{4S}{9X_1}\right)^4}}\right)$$

**Production store after percolation:**
$$S_{final} = S - Perc$$

**Total effective rainfall:**
$$P_R = P_R + Perc$$

---

### 5.3 Unit Hydrographs

The effective rainfall is split and routed through two unit hydrographs:

**Rainfall split:**
$$P_{UH1} = b \cdot P_R = 0.9 \cdot P_R$$
$$P_{UH2} = (1-b) \cdot P_R = 0.1 \cdot P_R$$

#### S-Curve Functions

**UH1 S-curve (faster response):**
$$SS_1(i) = \begin{cases}
0 & \text{if } i \leq 0 \\
\left(\frac{i}{X_4}\right)^D & \text{if } 0 < i < X_4 \\
1 & \text{if } i \geq X_4
\end{cases}$$

**UH2 S-curve (slower response):**
$$SS_2(i) = \begin{cases}
0 & \text{if } i \leq 0 \\
\frac{1}{2}\left(\frac{i}{X_4}\right)^D & \text{if } 0 < i \leq X_4 \\
1 - \frac{1}{2}\left(2 - \frac{i}{X_4}\right)^D & \text{if } X_4 < i < 2X_4 \\
1 & \text{if } i \geq 2X_4
\end{cases}$$

Where **D = 2.5** (fixed exponent).

#### Unit Hydrograph Ordinates

The ordinates are computed as successive differences:
$$UH_1(i) = SS_1(i) - SS_1(i-1)$$
$$UH_2(i) = SS_2(i) - SS_2(i-1)$$

#### Convolution

For each time step, the UH states are updated:

**UH1 convolution:**
$$\text{For } k = 1 \text{ to } \min(NH-1, \lfloor X_4 \rfloor + 1):$$
$$StUH_1(k) = StUH_1(k+1) + UH_1(k) \cdot P_{UH1}$$
$$StUH_1(NH) = UH_1(NH) \cdot P_{UH1}$$

**UH2 convolution:**
$$\text{For } k = 1 \text{ to } \min(2 \cdot NH-1, 2 \cdot \lfloor X_4 \rfloor + 1):$$
$$StUH_2(k) = StUH_2(k+1) + UH_2(k) \cdot P_{UH2}$$
$$StUH_2(2 \cdot NH) = UH_2(2 \cdot NH) \cdot P_{UH2}$$

Where **NH = 20** days.

---

### 5.4 Groundwater Exchange

The potential inter-catchment exchange is:

$$F = X_2 \cdot \left(\frac{R}{X_3} - X_5\right)$$

Where:
- R = current routing store level
- X3 = routing store capacity
- X5 = exchange threshold

**Physical interpretation:**
- F > 0: water import into the catchment (adds to stores)
- F < 0: water export from the catchment (removes from stores)

**Sign convention:** The direction of exchange depends on **both** X2 and the routing store level:
- When X2 > 0 and R/X3 > X5: positive F (import)
- When X2 > 0 and R/X3 < X5: negative F (export)
- When X2 < 0: signs reverse (X2 < 0 indicates net exporter catchment)

---

### 5.5 Routing Store

The routing store receives water from UH1 and groundwater exchange:

**Routing store inflow:**
$$R_{new} = R + (1-c) \cdot StUH_1(1) + F$$

Where **c = 0.4** (40% goes to exponential store, 60% to routing store).

**Actual exchange (with non-negativity constraint):**
$$F_{routing} = \begin{cases}
F & \text{if } R_{new} \geq 0 \\
-(R + (1-c) \cdot StUH_1(1)) & \text{if } R_{new} < 0
\end{cases}$$

$$R = \max(R_{new}, 0)$$

**Routing store outflow (non-linear reservoir):**
$$Q_R = R \cdot \left(1 - \frac{1}{\sqrt[4]{1 + \left(\frac{R}{X_3}\right)^4}}\right)$$

**Routing store update:**
$$R_{final} = R - Q_R$$

---

### 5.6 Exponential Store

The exponential store (unique to GR6J) provides additional slow baseflow.

> **Note:** Unlike the production and routing stores, the exponential store level can become **negative**. This is intentional and represents a deficit state that must be replenished before significant outflow occurs. The softplus formulation ensures outflow remains positive even for negative store values.

**Exponential store inflow:**
$$Exp_{new} = Exp + c \cdot StUH_1(1) + F$$

Where **c = 0.4**.

**Scaled store level (with numerical safeguards):**
$$AR = \text{clip}\left(\frac{Exp}{X_6}, -33, 33\right)$$

**Exponential store outflow (softplus-like function):**
$$Q_{RExp} = \begin{cases}
Exp + \frac{X_6}{e^{AR}} & \text{if } AR > 7 \\
X_6 \cdot e^{AR} & \text{if } AR < -7 \\
X_6 \cdot \ln(e^{AR} + 1) & \text{otherwise}
\end{cases}$$

The normal-range formula is the **softplus function**:
$$Q_{RExp} = X_6 \cdot \text{softplus}\left(\frac{Exp}{X_6}\right) = X_6 \cdot \ln\left(1 + e^{Exp/X_6}\right)$$

**Exponential store update:**
$$Exp_{final} = Exp - Q_{RExp}$$

---

### 5.7 Direct Branch

The direct branch routes water through UH2 with groundwater exchange:

**Direct branch outflow:**
$$Q_D = \max(StUH_2(1) + F, 0)$$

**Actual exchange from direct branch:**
$$F_{direct} = \begin{cases}
F & \text{if } StUH_2(1) + F \geq 0 \\
-StUH_2(1) & \text{if } StUH_2(1) + F < 0
\end{cases}$$

---

### 5.8 Total Streamflow

The total simulated streamflow is the sum of all outflow components:

$$Q = Q_R + Q_D + Q_{RExp}$$

With non-negativity constraint:
$$Q = \max(Q, 0)$$

---

## 6. Complete Algorithm

```
INPUT: P (precipitation), E (potential ET)
STATE: S (production), R (routing), Exp (exponential), StUH1, StUH2

FOR each time step:

  1. PRODUCTION STORE
     IF P < E:
       En = E - P
       WS = min(En/X1, 13)
       TWS = tanh(WS)
       Sr = S/X1
       ES = S * (2-Sr) * TWS / (1 + (1-Sr) * TWS)
       AE = ES + P
       S = S - ES
       Pn = 0
       PR = 0
     ELSE:
       AE = E
       Pn = P - E
       WS = min(Pn/X1, 13)
       TWS = tanh(WS)
       Sr = S/X1
       PS = X1 * (1 - Sr^2) * TWS / (1 + Sr * TWS)
       PR = Pn - PS
       S = S + PS
     ENDIF

  2. PERCOLATION
     S = max(S, 0)
     Sr4 = (S/X1)^4
     Perc = S * (1 - 1/sqrt(sqrt(1 + Sr4/25.62890625)))
     S = S - Perc
     PR = PR + Perc

  3. UNIT HYDROGRAPH SPLIT
     PRUH1 = 0.9 * PR
     PRUH2 = 0.1 * PR

  4. UH CONVOLUTION
     Convolve StUH1 with PRUH1
     Convolve StUH2 with PRUH2

  5. GROUNDWATER EXCHANGE
     F = X2 * (R/X3 - X5)

  6. ROUTING STORE
     R = R + 0.6 * StUH1(1) + F
     R = max(R, 0)
     Rr4 = (R/X3)^4
     QR = R * (1 - 1/sqrt(sqrt(1 + Rr4)))
     R = R - QR

  7. EXPONENTIAL STORE
     Exp = Exp + 0.4 * StUH1(1) + F
     AR = clip(Exp/X6, -33, 33)
     IF AR > 7:
       QRExp = Exp + X6/exp(AR)
     ELSEIF AR < -7:
       QRExp = X6 * exp(AR)
     ELSE:
       QRExp = X6 * ln(exp(AR) + 1)
     ENDIF
     Exp = Exp - QRExp

  8. DIRECT BRANCH
     QD = max(StUH2(1) + F, 0)

  9. TOTAL FLOW
     Q = QR + QD + QRExp
     Q = max(Q, 0)

OUTPUT: Q (streamflow in mm/day)
```

---

## 7. Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| b | 0.9 | Fraction of PR to UH1 (slow branch) |
| 1-b | 0.1 | Fraction of PR to UH2 (fast branch) |
| c | 0.4 | Fraction of UH1 output to exponential store |
| 1-c | 0.6 | Fraction of UH1 output to routing store |
| D | 2.5 | Exponent for unit hydrograph S-curves |
| NH | 20 | Length of UH1 (days) |
| 2*NH | 40 | Length of UH2 (days) |
| 9/4 | 2.25 | Percolation scaling factor |
| (9/4)^4 | 25.62890625 | Pre-computed percolation constant |
| 13 | - | Maximum scaled value for tanh (numerical safeguard) |
| 33 | - | Maximum AR for exponential store (numerical safeguard) |
| 7 | - | Threshold for exponential store branch equations |

---

## 8. Model Outputs

The model produces the following outputs at each time step:

| Output | Symbol | Description | Unit |
|--------|--------|-------------|------|
| PE | E | Potential evapotranspiration | mm/day |
| Precip | P | Precipitation | mm/day |
| Prod | S | Production store level | mm |
| Pn | Pn | Net rainfall | mm/day |
| Ps | PS | Storage infiltration | mm/day |
| AE | AE | Actual evapotranspiration | mm/day |
| Perc | Perc | Percolation | mm/day |
| PR | PR | Effective rainfall | mm/day |
| Q9 | StUH1(1) | Outflow from UH1 | mm/day |
| Q1 | StUH2(1) | Outflow from UH2 | mm/day |
| Rout | R | Routing store level | mm |
| Exch | F | Potential exchange | mm/day |
| AExch1 | F_routing | Actual exchange (routing) | mm/day |
| AExch2 | F_direct | Actual exchange (direct) | mm/day |
| AExch | - | Total actual exchange | mm/day |
| QR | QR | Routing store outflow | mm/day |
| QRExp | QRExp | Exponential store outflow | mm/day |
| Exp | Exp | Exponential store level | mm |
| QD | QD | Direct branch outflow | mm/day |
| Qsim | Q | Total simulated streamflow | mm/day |

---

## 9. Key Differences: GR6J vs GR4J

| Feature | GR4J | GR6J |
|---------|------|------|
| Parameters | 4 (X1-X4) | 6 (X1-X6) |
| Stores | 2 (Production, Routing) | 3 (Production, Routing, Exponential) |
| Slow baseflow | Single routing store | Routing + Exponential stores |
| Low-flow simulation | Limited | Improved (exponential store) |
| Exchange mechanism | Basic | Enhanced with threshold (X5) |

### Why GR6J?

The exponential store in GR6J was added specifically to:
1. **Improve low-flow simulation** during dry periods
2. **Better represent slow groundwater processes**
3. **Provide more flexibility** in recession curve fitting
4. **Reduce systematic bias** in baseflow prediction

---

## 10. References

### Primary Reference

> Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T. and Andréassian, V. (2011).
> **A downward structural sensitivity analysis of hydrological models to improve low-flow simulation.**
> *Journal of Hydrology*, 411(1-2), 66-76.
> doi: [10.1016/j.jhydrol.2011.09.034](https://doi.org/10.1016/j.jhydrol.2011.09.034)

### Related Publications

- Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology*, 279(1-4), 275-289.

- Le Moine, N. (2008). *Le bassin versant de surface vu par le souterrain: une voie d'amélioration des performances et du réalisme des modèles pluie-débit?* PhD thesis, Université Pierre et Marie Curie, Paris.

### Software Implementations

- **airGR** (R): Official implementation by INRAE
  https://gitlab.irstea.fr/HYCAR-Hydro/airgr

- **gr6j** (Rust/Python): Community implementation by S. Simoncelli
  https://github.com/s-simoncelli/GR6J

---

## Appendix A: Fortran Variable Mapping

For users referencing the official airGR Fortran implementation (`frun_GR6J.f90`), this table maps documentation symbols to Fortran variable names:

### State Variables

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| S | St(1) | Production store level |
| R | St(2) | Routing store level |
| Exp | St(3) | Exponential store level |
| StUH1 | StUH1(1:NH) | UH1 state array (NH=20) |
| StUH2 | StUH2(1:2*NH) | UH2 state array (2*NH=40) |

### Model Parameters

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| X1 | Param(1) | Production store capacity |
| X2 | Param(2) | Exchange coefficient |
| X3 | Param(3) | Routing store capacity |
| X4 | Param(4) | UH time constant |
| X5 | Param(5) | Exchange threshold |
| X6 | Param(6) | Exponential store scale |

### MISC Output Array

The Fortran implementation stores outputs in a `MISC(1:30)` array:

| Index | Name | Description |
|-------|------|-------------|
| MISC(1) | PE | Potential evapotranspiration |
| MISC(2) | Precip | Precipitation |
| MISC(3) | Prod | Production store level |
| MISC(4) | Pn | Net rainfall |
| MISC(5) | Ps | Storage infiltration |
| MISC(6) | AE | Actual evapotranspiration |
| MISC(7) | Perc | Percolation |
| MISC(8) | PR | Effective rainfall |
| MISC(9) | Q9 | UH1 outflow |
| MISC(10) | Q1 | UH2 outflow |
| MISC(11) | Rout | Routing store level |
| MISC(12) | Exch | Potential exchange (F) |
| MISC(13) | AExch1 | Actual exchange (routing) |
| MISC(14) | AExch2 | Actual exchange (direct) |
| MISC(15) | AExch | Total actual exchange (AExch1 + AExch2 + Exch) |
| MISC(16) | QR | Routing store outflow |
| MISC(17) | QRExp | Exponential store outflow |
| MISC(18) | Exp | Exponential store level |
| MISC(19) | QD | Direct branch outflow |
| MISC(20) | Qsim | Total simulated streamflow |

---

## Appendix B: Symbol Reference

### Primary Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| P | Daily precipitation | mm/day |
| E | Potential evapotranspiration | mm/day |
| Q | Simulated streamflow | mm/day |
| S | Production store level | mm |
| R | Routing store level | mm |
| Exp | Exponential store level | mm |

### Model Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| X1 | Production store capacity | mm |
| X2 | Exchange coefficient | mm/day |
| X3 | Routing store capacity | mm |
| X4 | UH time constant | days |
| X5 | Exchange threshold | - |
| X6 | Exponential store scale parameter | mm |

### Intermediate Variables (Production Store)

| Symbol | Description | Unit |
|--------|-------------|------|
| En | Net evapotranspiration (when P < E) | mm/day |
| Pn | Net rainfall (when P ≥ E) | mm/day |
| WS | Scaled value for tanh calculation | - |
| Sr | Store ratio (S/X1) | - |
| ES | Evaporation from production store | mm/day |
| PS | Storage infiltration | mm/day |
| Ea | Actual evapotranspiration | mm/day |
| PR | Effective rainfall (after percolation) | mm/day |
| Perc | Percolation from production store | mm/day |

### Intermediate Variables (Routing)

| Symbol | Description | Unit |
|--------|-------------|------|
| F | Groundwater exchange (potential) | mm/day |
| F_routing | Actual exchange at routing store | mm/day |
| F_direct | Actual exchange at direct branch | mm/day |
| QR | Routing store outflow | mm/day |
| QD | Direct branch outflow | mm/day |
| QRExp | Exponential store outflow | mm/day |
| AR | Scaled exponential store level (Exp/X6) | - |

### Unit Hydrograph Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| UH1 | Unit hydrograph 1 ordinates | - |
| UH2 | Unit hydrograph 2 ordinates | - |
| SS1 | S-curve for UH1 | - |
| SS2 | S-curve for UH2 | - |
| StUH1 | UH1 state array | mm |
| StUH2 | UH2 state array | mm |
| P_UH1 | Input to UH1 (0.9 × PR) | mm/day |
| P_UH2 | Input to UH2 (0.1 × PR) | mm/day |

### Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| b | 0.9 | UH split fraction (to UH1) |
| c | 0.4 | Routing split fraction (to exponential store) |
| D | 2.5 | S-curve exponent |
| NH | 20 | Unit hydrograph length (days) |

---

*Document generated from analysis of airGR (Fortran) and s-simoncelli/GR6J (Rust) implementations.*
