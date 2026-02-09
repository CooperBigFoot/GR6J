"""Tests for GR6J Rust backend equivalence.

Tests verify that the Rust-backed GR6J functions produce correct and
consistent results between step-by-step and full run execution.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, Parameters, run
from pydrology.models.gr6j.run import step
from pydrology.models.gr6j.types import State
from pydrology.models.gr6j.unit_hydrographs import compute_uh_ordinates


class TestStepRunEquivalence:
    """Tests that step-by-step execution matches run()."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)

    @pytest.fixture
    def state(self, params: Parameters) -> State:
        return State.initialize(params)

    def test_single_step_produces_valid_output(self, params: Parameters, state: State) -> None:
        """Single step produces valid, finite outputs."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)
        precip, pet = 10.0, 3.0

        new_state, fluxes = step(state, params, precip, pet, uh1_ord, uh2_ord)

        assert np.isfinite(fluxes["streamflow"])
        assert np.isfinite(fluxes["production_store"])
        assert np.isfinite(fluxes["routing_store"])
        assert np.isfinite(fluxes["exponential_store"])

    def test_multiple_steps_state_evolution(self, params: Parameters, state: State) -> None:
        """State evolves correctly over multiple steps."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

        current_state = state
        for i in range(10):
            precip = 5.0 + i * 2
            pet = 2.0 + i * 0.5
            current_state, _ = step(current_state, params, precip, pet, uh1_ord, uh2_ord)

        assert np.isfinite(current_state.production_store)
        assert np.isfinite(current_state.routing_store)
        assert np.isfinite(current_state.exponential_store)

    def test_step_all_output_fields(self, params: Parameters, state: State) -> None:
        """All 20 output fields are present and finite."""
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)
        precip, pet = 15.0, 4.0

        _, fluxes = step(state, params, precip, pet, uh1_ord, uh2_ord)

        expected_keys = [
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

        for key in expected_keys:
            assert key in fluxes, f"Missing key: {key}"
            assert np.isfinite(fluxes[key]), f"Non-finite value for {key}"


class TestRunEquivalence:
    """Tests that run() matches step-by-step execution."""

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

    def test_full_run_matches_step_loop(self, params: Parameters, forcing: ForcingData) -> None:
        """Full simulation produces identical streamflow to step-by-step."""
        result = run(params, forcing)

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
        """All 20 GR6J output fields match between run() and step loop."""
        result = run(params, forcing)

        state = State.initialize(params)
        uh1_ord, uh2_ord = compute_uh_ordinates(params.x4)

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

        for k in outputs_py:
            np.testing.assert_allclose(
                getattr(result.fluxes, k),
                outputs_py[k],
                rtol=1e-10,
                err_msg=f"Field {k} does not match",
            )


class TestEdgeCases:
    """Tests for edge cases in GR6J implementation."""

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
            precip=np.full(30, 1.0),
            pet=np.full(30, 8.0),
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
    """Tests for numerical stability."""

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

        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(np.isfinite(result.fluxes.production_store))
        assert np.all(np.isfinite(result.fluxes.routing_store))
        assert np.all(np.isfinite(result.fluxes.exponential_store))
        assert np.all(result.fluxes.streamflow >= 0)

    def test_extreme_parameter_combinations(self) -> None:
        """Various extreme parameter combinations produce valid outputs."""
        param_sets = [
            Parameters(x1=100, x2=-5, x3=20, x4=0.5, x5=0.5, x6=2),
            Parameters(x1=2500, x2=5, x3=500, x4=15, x5=0.5, x6=60),
            Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5),
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
