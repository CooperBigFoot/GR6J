"""Tests for GR2M Numba acceleration.

Tests verify that Numba-compiled functions produce identical results
to the pure Python implementations.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, Resolution
from pydrology.models.gr2m import Parameters, State, run, step
from pydrology.models.gr2m.run import _run_numba, _step_numba


class TestStepNumbaEquivalence:
    """Tests that _step_numba produces same results as step()."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(x1=500, x2=1.0)

    @pytest.fixture
    def state(self, params: Parameters) -> State:
        return State.initialize(params)

    def test_single_step_equivalence(self, params: Parameters, state: State) -> None:
        """Single step produces identical results in Numba and Python."""
        precip, pet = 80.0, 30.0

        # Python step
        new_state_py, fluxes_py = step(state, params, precip, pet)

        # Numba step
        state_arr = np.asarray(state).copy()
        params_arr = np.asarray(params)
        output_arr = np.zeros(11)
        _step_numba(state_arr, params_arr, precip, pet, output_arr)

        # Compare outputs
        assert output_arr[10] == pytest.approx(fluxes_py["streamflow"], rel=1e-10)
        assert output_arr[2] == pytest.approx(fluxes_py["production_store"], rel=1e-10)
        assert output_arr[8] == pytest.approx(fluxes_py["routing_store"], rel=1e-10)

    def test_multiple_steps_state_evolution(self, params: Parameters, state: State) -> None:
        """State evolves identically over multiple steps."""
        # Numba path
        state_arr = np.asarray(state).copy()
        params_arr = np.asarray(params)
        output_arr = np.zeros(11)

        # Python path
        state_py = State.initialize(params)

        for i in range(12):
            precip = 40.0 + i * 10
            pet = 20.0 + i * 5

            state_py, _ = step(state_py, params, precip, pet)
            _step_numba(state_arr, params_arr, precip, pet, output_arr)

        # Compare final states
        assert state_arr[0] == pytest.approx(state_py.production_store, rel=1e-10)
        assert state_arr[1] == pytest.approx(state_py.routing_store, rel=1e-10)

    def test_step_all_output_fields(self, params: Parameters, state: State) -> None:
        """All 11 output fields match between Numba and Python."""
        precip, pet = 100.0, 40.0

        # Python step
        _, fluxes_py = step(state, params, precip, pet)

        # Numba step
        state_arr = np.asarray(state).copy()
        params_arr = np.asarray(params)
        output_arr = np.zeros(11)
        _step_numba(state_arr, params_arr, precip, pet, output_arr)

        # Map output array indices to flux keys
        output_mapping = [
            "pet",
            "precip",
            "production_store",
            "rainfall_excess",
            "storage_fill",
            "actual_et",
            "percolation",
            "routing_input",
            "routing_store",
            "exchange",
            "streamflow",
        ]

        for idx, key in enumerate(output_mapping):
            assert output_arr[idx] == pytest.approx(fluxes_py[key], rel=1e-10), (
                f"Field {key} (index {idx}) does not match"
            )


class TestRunNumbaEquivalence:
    """Tests that _run_numba produces same results as the slow path."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(x1=500, x2=1.0)

    @pytest.fixture
    def forcing(self) -> ForcingData:
        n = 100
        rng = np.random.default_rng(42)
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="MS").values,
            precip=rng.uniform(20, 150, n),
            pet=rng.uniform(20, 120, n),
            resolution=Resolution.monthly,
        )

    def test_full_run_equivalence(self, params: Parameters, forcing: ForcingData) -> None:
        """Full simulation produces identical streamflow."""
        # Run via the normal API (uses Numba internally)
        result = run(params, forcing)

        # Run manually with the slow Python path for comparison
        state = State.initialize(params)

        streamflow_py = []
        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
            )
            streamflow_py.append(fluxes["streamflow"])

        np.testing.assert_allclose(result.fluxes.streamflow, streamflow_py, rtol=1e-10)

    def test_all_outputs_match(self, params: Parameters, forcing: ForcingData) -> None:
        """All 11 GR2M output fields match between Numba and Python paths."""
        result = run(params, forcing)

        state = State.initialize(params)

        # Collect all outputs from Python path
        outputs_py: dict[str, list[float]] = {
            k: []
            for k in [
                "pet",
                "precip",
                "production_store",
                "rainfall_excess",
                "storage_fill",
                "actual_et",
                "percolation",
                "routing_input",
                "routing_store",
                "exchange",
                "streamflow",
            ]
        }

        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
            )
            for k in outputs_py:
                outputs_py[k].append(fluxes[k])

        # Compare all fields
        for k in outputs_py:
            np.testing.assert_allclose(
                getattr(result.fluxes, k),
                outputs_py[k],
                rtol=1e-10,
                err_msg=f"Field {k} does not match",
            )

    def test_run_numba_direct(self, params: Parameters, forcing: ForcingData) -> None:
        """Test _run_numba directly against Python path."""
        state = State.initialize(params)

        # Numba path
        state_arr = np.asarray(state).copy()
        params_arr = np.asarray(params)
        outputs_arr = np.zeros((len(forcing), 11), dtype=np.float64)

        _run_numba(
            state_arr,
            params_arr,
            forcing.precip.astype(np.float64),
            forcing.pet.astype(np.float64),
            outputs_arr,
        )

        # Python path
        state_py = State.initialize(params)
        streamflow_py = []
        for i in range(len(forcing)):
            state_py, fluxes = step(
                state_py,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
            )
            streamflow_py.append(fluxes["streamflow"])

        np.testing.assert_allclose(outputs_arr[:, 10], streamflow_py, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases in Numba implementation."""

    def test_zero_precipitation(self) -> None:
        """Handles zero precipitation correctly."""
        params = Parameters(x1=500, x2=1.0)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.zeros(12),
            pet=np.full(12, 50.0),
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)

    def test_extreme_precipitation(self) -> None:
        """Handles extreme precipitation correctly."""
        params = Parameters(x1=500, x2=1.0)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.array([0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            pet=np.full(12, 50.0),
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_high_x2_water_gain(self) -> None:
        """Handles high X2 (water gain) correctly."""
        params = Parameters(x1=500, x2=2.0)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.full(12, 80.0),
            pet=np.full(12, 50.0),
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)

    def test_low_x2_water_loss(self) -> None:
        """Handles low X2 (water loss) correctly."""
        params = Parameters(x1=500, x2=0.2)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.full(12, 80.0),
            pet=np.full(12, 50.0),
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)

    def test_high_pet_dry_conditions(self) -> None:
        """Handles high PET with low precipitation correctly."""
        params = Parameters(x1=500, x2=1.0)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.full(12, 10.0),   # Very low precip
            pet=np.full(12, 150.0),     # High PET
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))


class TestNumericalStability:
    """Tests for numerical stability of Numba implementations."""

    def test_long_simulation_stability(self) -> None:
        """Long simulation maintains numerical stability."""
        params = Parameters(x1=500, x2=1.0)
        n = 500  # ~40 years of monthly data
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("1980-01-01", periods=n, freq="MS").values,
            precip=rng.uniform(20, 200, n),
            pet=rng.uniform(20, 150, n),
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)

        # All outputs should be finite
        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(np.isfinite(result.fluxes.production_store))
        assert np.all(np.isfinite(result.fluxes.routing_store))

        # Streamflow should be non-negative
        assert np.all(result.fluxes.streamflow >= 0)

    def test_tanh_safeguard(self) -> None:
        """Very high inputs don't cause tanh overflow."""
        params = Parameters(x1=100, x2=1.0)  # Small X1 to trigger safeguard
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
            precip=np.full(12, 5000.0),   # Very high precip
            pet=np.full(12, 3000.0),      # Very high PET
            resolution=Resolution.monthly,
        )
        result = run(params, forcing)

        # All outputs should be finite
        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(np.isfinite(result.fluxes.production_store))

    def test_extreme_parameter_combinations(self) -> None:
        """Various extreme parameter combinations produce valid outputs."""
        param_sets = [
            Parameters(x1=1, x2=0.2),     # Minimum-ish values
            Parameters(x1=2500, x2=2.0),  # Maximum values
            Parameters(x1=500, x2=1.0),   # Typical values
            Parameters(x1=100, x2=1.5),   # Small store, high exchange
            Parameters(x1=2000, x2=0.5),  # Large store, low exchange
        ]

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=24, freq="MS").values,
            precip=np.random.default_rng(42).uniform(20, 150, 24),
            pet=np.random.default_rng(42).uniform(20, 120, 24),
            resolution=Resolution.monthly,
        )

        for params in param_sets:
            result = run(params, forcing)
            assert np.all(np.isfinite(result.fluxes.streamflow)), f"Non-finite output for {params}"
            assert np.all(result.fluxes.streamflow >= 0), f"Negative streamflow for {params}"
