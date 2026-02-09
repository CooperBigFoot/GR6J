"""Tests for GR2M Rust backend.

Tests verify edge cases and numerical stability of the Rust-backed
run() and step() functions.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, Resolution
from pydrology.models.gr2m import Parameters, State, run, step


class TestRunStepConsistency:
    """Tests that run() and step() produce consistent results."""

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

    def test_full_run_matches_step_loop(self, params: Parameters, forcing: ForcingData) -> None:
        """Full simulation via run() matches sequential step() calls."""
        result = run(params, forcing)

        state = State.initialize(params)

        streamflow_step = []
        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
            )
            streamflow_step.append(fluxes["streamflow"])

        np.testing.assert_allclose(result.fluxes.streamflow, streamflow_step, rtol=1e-10)

    def test_all_outputs_match(self, params: Parameters, forcing: ForcingData) -> None:
        """All 11 GR2M output fields match between run() and step() loop."""
        result = run(params, forcing)

        state = State.initialize(params)

        outputs_step: dict[str, list[float]] = {
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
            for k in outputs_step:
                outputs_step[k].append(fluxes[k])

        for k in outputs_step:
            np.testing.assert_allclose(
                getattr(result.fluxes, k),
                outputs_step[k],
                rtol=1e-10,
                err_msg=f"Field {k} does not match",
            )


class TestEdgeCases:
    """Tests for edge cases in GR2M implementation."""

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
    """Tests for numerical stability."""

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
