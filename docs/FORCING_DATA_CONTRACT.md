# ForcingData Validation Contract

This document specifies the validation rules and data contract for the `ForcingData` class.

## Overview

`ForcingData` is a Pydantic-validated container for hydrological forcing data. It enforces
strict validation rules to ensure data quality and prevent runtime errors.

## Fields

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `time` | `np.ndarray[datetime64]` | Yes | - | Timestamp for each observation |
| `precip` | `np.ndarray[float64]` | Yes | mm/day | Daily precipitation |
| `pet` | `np.ndarray[float64]` | Yes | mm/day | Daily potential evapotranspiration |
| `temp` | `np.ndarray[float64]` | No* | deg C | Daily mean air temperature |

*Required when using the GR6J-CemaNeige coupled model (`pydrology.models.gr6j_cemaneige`)

## Validation Rules

### 1. Array Dimensionality

All arrays must be 1-dimensional.

**Valid:**

```python
precip = np.array([1.0, 2.0, 3.0])  # Shape: (3,)
```

**Invalid:**

```python
precip = np.array([[1.0], [2.0]])   # Shape: (2, 1) - 2D array
```

**Error:** `ValidationError: precip array must be 1D, got 2D`

### 2. Length Consistency

All provided arrays must have the same length as the `time` array.

**Valid:**

```python
time = np.array(['2020-01-01', '2020-01-02'], dtype='datetime64')
precip = np.array([1.0, 2.0])  # Same length as time
```

**Invalid:**

```python
time = np.array(['2020-01-01', '2020-01-02'], dtype='datetime64')
precip = np.array([1.0, 2.0, 3.0])  # Different length
```

**Error:** `ValidationError: precip length 3 does not match time length 2`

### 3. NaN Rejection

NaN values are not allowed in any numeric array (precip, pet, temp).

**Invalid:**

```python
precip = np.array([1.0, np.nan, 3.0])
```

**Error:** `ValidationError: precip array contains NaN values`

**Why:** NaN values in forcing data typically indicate data quality issues. The model
requires explicit handling of missing data before simulation.

### 4. Type Coercion

**Numeric arrays:** Automatically coerced to `float64`

```python
# Input as integers
precip = np.array([1, 2, 3])  # int64
# Stored as float64
assert forcing.precip.dtype == np.float64
```

**Time array:** Automatically coerced to `datetime64[ns]`

```python
# Input as strings or datetime objects
time = np.array(['2020-01-01', '2020-01-02'])
# Stored as datetime64[ns]
assert forcing.time.dtype == np.dtype('datetime64[ns]')
```

## Immutability

ForcingData is frozen (immutable) after creation:

```python
forcing = ForcingData(...)
forcing.precip = new_array  # Raises ValidationError
```

## Creating ForcingData

### From NumPy Arrays

```python
import numpy as np
from pydrology import ForcingData

forcing = ForcingData(
    time=np.arange(365, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.random.uniform(0, 20, 365),
    pet=np.random.uniform(1, 5, 365),
)
```

### From Pandas DataFrame

```python
import pandas as pd
from pydrology import ForcingData

df = pd.read_csv('forcing.csv', parse_dates=['date'])

forcing = ForcingData(
    time=df['date'].values,
    precip=df['precip'].values,
    pet=df['pet'].values,
    temp=df['temp'].values if 'temp' in df else None,
)
```

### From CSV with pandas

```python
import pandas as pd
from pydrology import ForcingData

df = pd.read_csv('forcing.csv')

forcing = ForcingData(
    time=pd.to_datetime(df['date']).values,
    precip=df['precip'].to_numpy(),
    pet=df['pet'].to_numpy(),
)
```

### From NetCDF

```python
import xarray as xr
from pydrology import ForcingData

ds = xr.open_dataset('forcing.nc')

forcing = ForcingData(
    time=ds['time'].values,
    precip=ds['precipitation'].values,
    pet=ds['pet'].values,
)
```

## Error Handling

### ValidationError Structure

Pydantic validation errors contain detailed information:

```python
try:
    forcing = ForcingData(
        time=np.arange(3, dtype='datetime64[D]'),
        precip=np.array([1.0, np.nan, 3.0]),
        pet=np.array([1.0, 2.0, 3.0]),
    )
except ValidationError as e:
    print(e.errors())
    # [{'loc': ('precip',), 'msg': 'precip array contains NaN values', ...}]
```

### Handling Missing Data

Before creating ForcingData, handle missing values:

```python
import numpy as np
import pandas as pd

# Option 1: Forward fill
df['precip'] = df['precip'].ffill()

# Option 2: Interpolate
df['pet'] = df['pet'].interpolate()

# Option 3: Remove rows with missing values
df = df.dropna(subset=['precip', 'pet'])

# Now create ForcingData
forcing = ForcingData(
    time=df['date'].values,
    precip=df['precip'].values,
    pet=df['pet'].values,
)
```

## Best Practices

1. **Validate early:** Create ForcingData as early as possible in your pipeline to catch
   data issues immediately.

2. **Check lengths:** If loading from multiple sources, verify array lengths match before
   creating ForcingData.

3. **Handle NaN explicitly:** Decide on a strategy for missing data (interpolation,
   fill, or exclusion) before creating ForcingData.

4. **Use appropriate dtypes:** While coercion is automatic, providing data in the correct
   dtype (float64 for numerics, datetime64 for time) is more efficient.
