"""Tests for GR6J Numba acceleration.

Tests verify that Numba-compiled functions produce identical results
to the pure Python implementations.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, Parameters, run
from pydrology.models.gr6j.run import _run_numba, _step_numba, step
from pydrology.models.gr6j.types import State
from pydrology.models.gr6j.unit_hydrographs import compute_uh_ordinates


class TestStepNumbaEquivalence:
    """Tests that _step_numba produces same results as step()."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)

    @pytest.fixture
    def state(self, params: Parameters) -> State:
        return State.initialize(params)

    def test_single_step_equivalence(self, params: Parameters, state: State) -> None:
        """Single step produces identical results in Numba and Python."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)
        precip, pet = 10.0, 3.0

        # Python step
        new_state_py, fluxes_py = step(state, params, precip, pet, uh1_ord, uh2_ord)

        # Numba step
        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        output_arr = np.zeros(20)
        _step_numba(state_arr, params_arr, precip, pet, uh1_ord, uh2_ord, output_arr)

        # Compare outputs
        assert output_arr[19] == pytest.approx(fluxes_py["streamflow"], rel=1e-10)
        assert output_arr[2] == pytest.approx(fluxes_py["production_store"], rel=1e-10)
        assert output_arr[10] == pytest.approx(fluxes_py["routing_store"], rel=1e-10)
        assert output_arr[17] == pytest.approx(fluxes_py["exponential_store"], rel=1e-10)

    def test_multiple_steps_state_evolution(self, params: Parameters, state: State) -> None:
        """State evolves identically over multiple steps."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

        # Numba path
        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        output_arr = np.zeros(20)

        # Python path
        state_py = State.initialize(params)

        for i in range(10):
            precip = 5.0 + i * 2
            pet = 2.0 + i * 0.5

            state_py, _ = step(state_py, params, precip, pet, uh1_ord, uh2_ord)
            _step_numba(state_arr, params_arr, precip, pet, uh1_ord, uh2_ord, output_arr)

        # Compare final states
        assert state_arr[0] == pytest.approx(state_py.production_store, rel=1e-10)
        assert state_arr[1] == pytest.approx(state_py.routing_store, rel=1e-10)
        assert state_arr[2] == pytest.approx(state_py.exponential_store, rel=1e-10)

    def test_step_all_output_fields(self, params: Parameters, state: State) -> None:
        """All 20 output fields match between Numba and Python."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)
        precip, pet = 15.0, 4.0

        # Python step
        _, fluxes_py = step(state, params, precip, pet, uh1_ord, uh2_ord)

        # Numba step
        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        output_arr = np.zeros(20)
        _step_numba(state_arr, params_arr, precip, pet, uh1_ord, uh2_ord, output_arr)

        # Map output array indices to flux keys
        output_mapping = [
            "pet",
            "precip",
            "production_store",
            "net_rainfall",
            "storage_infiltration",
            "actual_et",
            "percolation",
            "effective_rainfall",
            "q9",
            "q1",
            "routing_store",
            "exchange",
            "actual_exchange_routing",
            "actual_exchange_direct",
            "actual_exchange_total",
            "qr",
            "qrexp",
            "exponential_store",
            "qd",
            "streamflow",
        ]

        for idx, key in enumerate(output_mapping):
            assert output_arr[idx] == pytest.approx(fluxes_py[key], rel=1e-10), (
                f"Field {key} (index {idx}) does not match"
            )

    def test_uh_state_evolution(self, params: Parameters, state: State) -> None:
        """Unit hydrograph states evolve correctly."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)
        precip, pet = 20.0, 3.0

        # Python step
        new_state_py, _ = step(state, params, precip, pet, uh1_ord, uh2_ord)

        # Numba step
        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        output_arr = np.zeros(20)
        _step_numba(state_arr, params_arr, precip, pet, uh1_ord, uh2_ord, output_arr)

        # Compare UH1 states (indices 3:23)
        np.testing.assert_allclose(state_arr[3:23], new_state_py.uh1_states, rtol=1e-10)
        # Compare UH2 states (indices 23:63)
        np.testing.assert_allclose(state_arr[23:63], new_state_py.uh2_states, rtol=1e-10)


class TestRunNumbaEquivalence:
    """Tests that _run_numba produces same results as the slow path."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)

    @pytest.fixture
    def forcing(self) -> ForcingData:
        n = 100
        rng = np.random.default_rng(42)
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 20, n),
            pet=rng.uniform(2, 6, n),
        )

    def test_full_run_equivalence(self, params: Parameters, forcing: ForcingData) -> None:
        """Full simulation produces identical streamflow."""
        # Run via the normal API (uses Numba internally)
        result = run(params, forcing)

        # Run manually with the slow Python path for comparison
        state = State.initialize(params)
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

        streamflow_py = []
        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
                uh1_ord,
                uh2_ord,
            )
            streamflow_py.append(fluxes["streamflow"])

        np.testing.assert_allclose(result.fluxes.streamflow, streamflow_py, rtol=1e-10)

    def test_all_outputs_match(self, params: Parameters, forcing: ForcingData) -> None:
        """All 20 GR6J output fields match between Numba and Python paths."""
        result = run(params, forcing)

        state = State.initialize(params)
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

        # Collect all outputs from Python path
        outputs_py: dict[str, list[float]] = {
            k: []
            for k in [
                "pet",
                "precip",
                "production_store",
                "net_rainfall",
                "storage_infiltration",
                "actual_et",
                "percolation",
                "effective_rainfall",
                "q9",
                "q1",
                "routing_store",
                "exchange",
                "actual_exchange_routing",
                "actual_exchange_direct",
                "actual_exchange_total",
                "qr",
                "qrexp",
                "exponential_store",
                "qd",
                "streamflow",
            ]
        }

        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
                uh1_ord,
                uh2_ord,
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
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

        # Numba path
        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        outputs_arr = np.zeros((len(forcing), 20), dtype=np.float64)

        _run_numba(
            state_arr,
            params_arr,
            forcing.precip.astype(np.float64),
            forcing.pet.astype(np.float64),
            uh1_ord,
            uh2_ord,
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
                uh1_ord,
                uh2_ord,
            )
            streamflow_py.append(fluxes["streamflow"])

        np.testing.assert_allclose(outputs_arr[:, 19], streamflow_py, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases in Numba implementation."""

    def test_zero_precipitation(self) -> None:
        """Handles zero precipitation correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.zeros(10),
            pet=np.full(10, 3.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)

    def test_extreme_precipitation(self) -> None:
        """Handles extreme precipitation correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.array([0, 0, 0, 200, 0, 0, 0, 0, 0, 0], dtype=float),
            pet=np.full(10, 3.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_negative_exchange_coefficient(self) -> None:
        """Handles negative X2 (export) correctly."""
        params = Parameters(x1=350, x2=-2.0, x3=90, x4=1.7, x5=0.5, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.full(10, 10.0),
            pet=np.full(10, 3.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)

    def test_very_small_x4(self) -> None:
        """Handles very small X4 (fast routing) correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=0.5, x5=0, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.full(10, 10.0),
            pet=np.full(10, 3.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_large_x4(self) -> None:
        """Handles large X4 (slow routing) correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=15.0, x5=0, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=50, freq="D").values,
            precip=np.full(50, 10.0),
            pet=np.full(50, 3.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_high_pet_dry_conditions(self) -> None:
        """Handles high PET with low precipitation correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=30, freq="D").values,
            precip=np.full(30, 1.0),  # Very low precip
            pet=np.full(30, 8.0),  # High PET
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_alternating_wet_dry(self) -> None:
        """Handles alternating wet and dry conditions correctly."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        n = 20
        precip = np.array([30.0, 0.0] * (n // 2))
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=precip,
            pet=np.full(n, 4.0),
        )
        result = run(params, forcing)
        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))


class TestNumericalStability:
    """Tests for numerical stability of Numba implementations."""

    def test_long_simulation_stability(self) -> None:
        """Long simulation maintains numerical stability."""
        params = Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)
        n = 1000
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 30, n),
            pet=rng.uniform(1, 8, n),
        )
        result = run(params, forcing)

        # All outputs should be finite
        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(np.isfinite(result.fluxes.production_store))
        assert np.all(np.isfinite(result.fluxes.routing_store))
        assert np.all(np.isfinite(result.fluxes.exponential_store))

        # Streamflow should be non-negative
        assert np.all(result.fluxes.streamflow >= 0)

    def test_extreme_parameter_combinations(self) -> None:
        """Various extreme parameter combinations produce valid outputs."""
        param_sets = [
            Parameters(x1=100, x2=-5, x3=20, x4=0.5, x5=0.5, x6=2),  # Small stores
            Parameters(x1=2500, x2=5, x3=500, x4=15, x5=0.5, x6=60),  # Large stores
            Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5),  # No exchange
        ]

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=50, freq="D").values,
            precip=np.random.default_rng(42).uniform(0, 30, 50),
            pet=np.random.default_rng(42).uniform(1, 8, 50),
        )

        for params in param_sets:
            result = run(params, forcing)
            assert np.all(np.isfinite(result.fluxes.streamflow)), f"Non-finite output for {params}"
            assert np.all(result.fluxes.streamflow >= 0), f"Negative streamflow for {params}"
