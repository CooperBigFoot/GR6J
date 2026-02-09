use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Validate that a numpy array is C-contiguous and return its slice.
pub fn contiguous_slice<'py>(arr: &'py PyReadonlyArray1<'py, f64>) -> PyResult<&'py [f64]> {
    arr.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("array must be C-contiguous")
    })
}

/// Validate length + contiguity of a numpy array.
pub fn checked_slice<'py>(
    arr: &'py PyReadonlyArray1<'py, f64>,
    expected_len: usize,
    name: &str,
) -> PyResult<&'py [f64]> {
    let slice = contiguous_slice(arr)?;
    if slice.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} must have {} elements, got {}",
            name, expected_len, slice.len()
        )));
    }
    Ok(slice)
}

/// Validate minimum length + contiguity of a numpy array.
pub fn checked_slice_min<'py>(
    arr: &'py PyReadonlyArray1<'py, f64>,
    min_len: usize,
    name: &str,
) -> PyResult<&'py [f64]> {
    let slice = contiguous_slice(arr)?;
    if slice.len() < min_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} must have at least {} elements, got {}",
            name, min_len, slice.len()
        )));
    }
    Ok(slice)
}
