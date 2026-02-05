# GR2M Model Technical Definition

> **Génie Rural à 2 paramètres Mensuel**
> A lumped conceptual rainfall-runoff model for monthly streamflow simulation

This document describes the GR2M model as implemented in airGR (`frun_GR2M.f90`).

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Model Parameters](#2-model-parameters)
3. [State Variables](#3-state-variables)
4. [Model Structure Diagram](#4-model-structure-diagram)
5. [Mathematical Equations](#5-mathematical-equations)
   - [5.1 Production Store - Rainfall](#51-production-store---rainfall)
   - [5.2 Production Store - Evaporation](#52-production-store---evaporation)
   - [5.3 Percolation](#53-percolation)
   - [5.4 Routing Store](#54-routing-store)
   - [5.5 Total Streamflow](#55-total-streamflow)
6. [Complete Algorithm](#6-complete-algorithm)
7. [Numerical Constants](#7-numerical-constants)
8. [Model Outputs](#8-model-outputs)
9. [References](#9-references)
10. [Appendix A: Fortran Variable Mapping](#appendix-a-fortran-variable-mapping)
11. [Appendix B: Symbol Reference](#appendix-b-symbol-reference)

---

## 1. Model Overview

**GR2M** (Génie Rural à 2 paramètres Mensuel) is a lumped, conceptual, monthly rainfall-runoff model developed by INRAE (formerly IRSTEA/Cemagref) in France. It is the simplest GR model with storage, featuring only two parameters and two stores.

### Key Characteristics

| Property | Value |
|----------|-------|
| Time step | Monthly |
| Spatial resolution | Lumped (catchment-scale) |
| Parameters | 2 calibrated parameters |
| Stores | 2 (Production, Routing) |
| Unit hydrographs | None |
| Inputs | Precipitation (P), Potential Evapotranspiration (E) |
| Output | Streamflow at catchment outlet (Q) |

### Model Philosophy

GR2M operates in **simulation mode** (concurrent prediction), meaning it does not use past observed streamflow as input. Its simplicity makes it ideal for:

- Data-scarce regions
- Long-term water balance studies
- Climate change impact assessments
- Regional hydrological modeling
- Preliminary catchment analysis

### Warm-up Period

The model requires a **warm-up period** to initialize internal states before producing reliable outputs.

| Aspect | Recommendation |
|--------|----------------|
| Duration | **12-24 months** minimum |
| Purpose | Allow stores to reach dynamic equilibrium |
| Outputs | Should be discarded from analysis |

---

## 2. Model Parameters

GR2M uses **2 calibrated parameters** that control different aspects of the rainfall-runoff transformation:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **X1** | Production store capacity | mm | [1, 2500] |
| **X2** | Groundwater exchange coefficient | - | [0.2, 2.0] |

### Physical Interpretation

- **X1 (Production store capacity)**: Controls the maximum soil moisture storage. Larger values mean more water can be retained in the catchment before generating runoff. Represents the combined storage capacity of soil, vegetation interception, and surface depressions.

- **X2 (Groundwater exchange coefficient)**: Controls the transformation in the routing store. Values > 1 indicate water gains (from neighboring catchments or deep aquifers); values < 1 indicate water losses. This parameter implicitly accounts for intercatchment groundwater exchanges.

---

## 3. State Variables

The model maintains **2 stores** with evolving water levels:

| Store | Symbol | Description | Initialization |
|-------|--------|-------------|----------------|
| Production store | S | Soil moisture reservoir | 30% of X1 |
| Routing store | R | Groundwater reservoir | 30% of X1 |

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
              │  • Neutralization      │
              │  • Evapotranspiration  │
              │  • Soil storage        │
              └────────────┬───────────┘
                           │
                     Percolation (P2)
                           │
                           ▼
                   ┌───────────────┐
         P1 ─────►│               │
      (Excess)    │  ROUTING      │
                  │   STORE       │
                  │               │
                  └───────┬───────┘
                          │
                    Exchange (X2)
                          │
                          ▼
                   ┌───────────────┐
                   │   ROUTING     │
                   │   EQUATION    │
                   │  Q = R²/(R+60)│
                   └───────┬───────┘
                           │
                           ▼
                        OUTPUT
                     Streamflow (Q)
```

---

## 5. Mathematical Equations

### 5.1 Production Store - Rainfall

The production store first receives and neutralizes precipitation:

**Scaled precipitation (with numerical safeguard):**
$$W_S = \min\left(\frac{P}{X_1}, 13\right)$$

**Hyperbolic tangent:**
$$\tanh(W_S) = \frac{e^{2W_S} - 1}{e^{2W_S} + 1}$$

**Neutralization - new store level after rainfall:**
$$S_1 = \frac{S + X_1 \cdot \tanh(W_S)}{1 + \frac{S}{X_1} \cdot \tanh(W_S)}$$

**Rainfall excess (water not stored):**
$$P_1 = P + S - S_1$$

**Storage fill:**
$$P_S = P - P_1$$

---

### 5.2 Production Store - Evaporation

After rainfall neutralization, evapotranspiration is extracted:

**Scaled evapotranspiration (with numerical safeguard):**
$$W_S = \min\left(\frac{E}{X_1}, 13\right)$$

**Store level after evaporation:**
$$S_2 = \frac{S_1 \cdot (1 - \tanh(W_S))}{1 + (1 - S_1/X_1) \cdot \tanh(W_S)}$$

**Actual evapotranspiration:**
$$AE = S_1 - S_2$$

**Production store update:**
$$S = S_2$$

---

### 5.3 Percolation

Water percolates from the production store to the routing store:

**Store ratio:**
$$S_r = \frac{S}{X_1}$$

**Percolation calculation:**
$$S_{final} = \frac{S}{\sqrt[3]{1 + S_r^3}}$$

**Percolation amount:**
$$P_2 = S - S_{final}$$

**Production store after percolation:**
$$S = S_{final}$$

**Total water to routing:**
$$P_3 = P_1 + P_2$$

---

### 5.4 Routing Store

The routing store receives water and applies groundwater exchange:

**Routing store inflow:**
$$R_1 = R + P_3$$

**Groundwater exchange:**
$$R_2 = X_2 \cdot R_1$$

**Actual exchange:**
$$AEXCH = R_2 - R_1$$

---

### 5.5 Total Streamflow

**Routing equation (quadratic reservoir):**
$$Q = \frac{R_2^2}{R_2 + 60}$$

**Routing store update:**
$$R = R_2 - Q$$

---

## 6. Complete Algorithm

```
INPUT: P (precipitation), E (potential ET)
STATE: S (production store), R (routing store)

FOR each month:

  1. PRODUCTION STORE - RAINFALL NEUTRALIZATION
     WS = P / X1
     IF WS > 13: WS = 13
     TWS = (exp(2*WS) - 1) / (exp(2*WS) + 1)
     Sr = S / X1
     S1 = (S + X1 * TWS) / (1 + Sr * TWS)
     P1 = P + S - S1
     PS = P - P1

  2. PRODUCTION STORE - EVAPORATION
     WS = E / X1
     IF WS > 13: WS = 13
     TWS = (exp(2*WS) - 1) / (exp(2*WS) + 1)
     Sr = S1 / X1
     S2 = S1 * (1 - TWS) / (1 + (1 - Sr) * TWS)
     AE = S1 - S2
     S = S2

  3. PERCOLATION
     Sr = S / X1
     Sr = Sr * Sr * Sr        ! Sr^3
     Sr = Sr + 1
     S = S / Sr^(1/3)
     P2 = S2 - S

  4. TOTAL WATER TO ROUTING
     P3 = P1 + P2

  5. ROUTING STORE
     R1 = R + P3
     R2 = X2 * R1
     AEXCH = R2 - R1

  6. STREAMFLOW
     Q = R2 * R2 / (R2 + 60)
     R = R2 - Q

OUTPUT: Q (streamflow in mm/month)
```

---

## 7. Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| tanh limit | 13.0 | Maximum scaled value for tanh (numerical safeguard) |
| Routing denominator | 60 | Constant in quadratic routing equation |
| Percolation exponent | 1/3 | Cube root in percolation formula |

---

## 8. Model Outputs

The model produces the following outputs at each time step:

| Output | Symbol | Description | Unit |
|--------|--------|-------------|------|
| PE | E | Potential evapotranspiration | mm/month |
| Precip | P | Precipitation | mm/month |
| Prod | S | Production store level | mm |
| Pn | P1 | Rainfall excess | mm/month |
| Ps | PS | Storage infiltration | mm/month |
| AE | AE | Actual evapotranspiration | mm/month |
| Perc | P2 | Percolation | mm/month |
| PR | P3 | Total routing input | mm/month |
| Rout | R | Routing store level | mm |
| Exch | AEXCH | Water exchange | mm/month |
| Qsim | Q | Total simulated streamflow | mm/month |

### Fortran MISC Array Mapping

| Index | Name | Description |
|-------|------|-------------|
| MISC(1) | PE | Potential evapotranspiration |
| MISC(2) | Precip | Precipitation |
| MISC(3) | Prod | Production store level |
| MISC(4) | Pn | Rainfall excess |
| MISC(5) | Ps | Storage fill |
| MISC(6) | AE | Actual evapotranspiration |
| MISC(7) | Perc | Percolation |
| MISC(8) | PR | Total routing input |
| MISC(9) | Rout | Routing store level |
| MISC(10) | Exch | Water exchange |
| MISC(11) | Qsim | Simulated discharge |

---

## 9. References

### Primary Reference

> Mouelhi, S., Michel, C., Perrin, C. and Andréassian, V. (2006).
> **Stepwise development of a two-parameter monthly water balance model.**
> *Journal of Hydrology*, 318(1-4), 200-214.
> doi: [10.1016/j.jhydrol.2005.06.014](https://doi.org/10.1016/j.jhydrol.2005.06.014)

### Related Publications

- Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology*, 279(1-4), 275-289.

- Mouelhi, S. (2003). Vers une chaîne cohérente de modèles pluie-débit conceptuels globaux aux pas de temps pluriannuel, annuel, mensuel et journalier. PhD thesis, ENGREF, Cemagref Antony.

### Software Implementation

- **airGR** (R): Official implementation by INRAE
  <https://gitlab.irstea.fr/HYCAR-Hydro/airgr>

---

## Appendix A: Fortran Variable Mapping

For users referencing the official airGR Fortran implementation (`frun_GR2M.f90`):

### State Variables

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| S | St(1) | Production store level |
| R | St(2) | Routing store level |

### Model Parameters

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| X1 | Param(1) | Production store capacity |
| X2 | Param(2) | Groundwater exchange coefficient |

### Internal Variables

| Documentation | Fortran | Description |
|---------------|---------|-------------|
| P | P1 | Input precipitation |
| E | E | Potential evapotranspiration |
| WS | WS | Scaled value for tanh |
| TWS | TWS | tanh(WS) result |
| S1 | S1 | Store after rainfall |
| S2 | S2 | Store after evaporation |
| P1 | P1 | Rainfall excess |
| P2 | P2 | Percolation |
| P3 | P3 | Total to routing |
| R1 | R1 | Routing store after inflow |
| R2 | R2 | Routing store after exchange |
| AE | AE | Actual evapotranspiration |
| Q | Q | Simulated discharge |

---

## Appendix B: Symbol Reference

### Primary Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| P | Monthly precipitation | mm/month |
| E | Potential evapotranspiration | mm/month |
| Q | Simulated streamflow | mm/month |
| S | Production store level | mm |
| R | Routing store level | mm |

### Model Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| X1 | Production store capacity | mm |
| X2 | Groundwater exchange coefficient | - |

### Intermediate Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| WS | Scaled value for tanh calculation | - |
| Sr | Store ratio (S/X1) | - |
| S1 | Production store after rainfall | mm |
| S2 | Production store after evaporation | mm |
| P1 | Rainfall excess | mm/month |
| P2 | Percolation | mm/month |
| P3 | Total water to routing | mm/month |
| PS | Storage fill | mm/month |
| AE | Actual evapotranspiration | mm/month |
| R1 | Routing store after inflow | mm |
| R2 | Routing store after exchange | mm |
| AEXCH | Actual groundwater exchange | mm/month |

---

*Document generated from analysis of airGR frun_GR2M.f90 Fortran source code.*
