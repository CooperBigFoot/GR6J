"""Tests for output dataclasses: GR6JOutput, SnowOutput, and ModelOutput.

Tests cover field counts, immutability, dictionary conversion, and DataFrame
generation for the structured output dataclasses.
"""

import dataclasses
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from gr6j import GR6JOutput, ModelOutput, SnowOutput


class TestGR6JOutput:
    """Tests for the GR6JOutput frozen dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """Test creation with all 20 flux fields."""
        n = 5
        output = GR6JOutput(
            pet=np.zeros(n),
            precip=np.zeros(n),
            production_store=np.zeros(n),
            net_rainfall=np.zeros(n),
            storage_infiltration=np.zeros(n),
            actual_et=np.zeros(n),
            percolation=np.zeros(n),
            effective_rainfall=np.zeros(n),
            q9=np.zeros(n),
            q1=np.zeros(n),
            routing_store=np.zeros(n),
            exchange=np.zeros(n),
            actual_exchange_routing=np.zeros(n),
            actual_exchange_direct=np.zeros(n),
            actual_exchange_total=np.zeros(n),
            qr=np.zeros(n),
            qrexp=np.zeros(n),
            exponential_store=np.zeros(n),
            qd=np.zeros(n),
            streamflow=np.zeros(n),
        )

        assert output.streamflow is not None
        assert len(output.streamflow) == n

    def test_to_dict_returns_all_fields(self) -> None:
        """Test to_dict contains all 20 fields."""
        n = 3
        output = GR6JOutput(
            pet=np.array([1.0, 2.0, 3.0]),
            precip=np.array([4.0, 5.0, 6.0]),
            production_store=np.zeros(n),
            net_rainfall=np.zeros(n),
            storage_infiltration=np.zeros(n),
            actual_et=np.zeros(n),
            percolation=np.zeros(n),
            effective_rainfall=np.zeros(n),
            q9=np.zeros(n),
            q1=np.zeros(n),
            routing_store=np.zeros(n),
            exchange=np.zeros(n),
            actual_exchange_routing=np.zeros(n),
            actual_exchange_direct=np.zeros(n),
            actual_exchange_total=np.zeros(n),
            qr=np.zeros(n),
            qrexp=np.zeros(n),
            exponential_store=np.zeros(n),
            qd=np.zeros(n),
            streamflow=np.array([7.0, 8.0, 9.0]),
        )

        result = output.to_dict()

        assert len(result) == 20
        assert "pet" in result
        assert "precip" in result
        assert "streamflow" in result
        np.testing.assert_array_equal(result["pet"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result["streamflow"], np.array([7.0, 8.0, 9.0]))

    def test_is_frozen(self) -> None:
        """Test output is immutable."""
        n = 5
        output = GR6JOutput(
            pet=np.zeros(n),
            precip=np.zeros(n),
            production_store=np.zeros(n),
            net_rainfall=np.zeros(n),
            storage_infiltration=np.zeros(n),
            actual_et=np.zeros(n),
            percolation=np.zeros(n),
            effective_rainfall=np.zeros(n),
            q9=np.zeros(n),
            q1=np.zeros(n),
            routing_store=np.zeros(n),
            exchange=np.zeros(n),
            actual_exchange_routing=np.zeros(n),
            actual_exchange_direct=np.zeros(n),
            actual_exchange_total=np.zeros(n),
            qr=np.zeros(n),
            qrexp=np.zeros(n),
            exponential_store=np.zeros(n),
            qd=np.zeros(n),
            streamflow=np.zeros(n),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            output.streamflow = np.ones(n)  # type: ignore[misc]

    def test_field_count_is_twenty(self) -> None:
        """Test there are exactly 20 flux fields."""
        field_count = len(fields(GR6JOutput))

        assert field_count == 20


class TestSnowOutput:
    """Tests for the SnowOutput frozen dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """Test creation with all 12 snow fields."""
        n = 5
        output = SnowOutput(
            precip_raw=np.zeros(n),
            snow_pliq=np.zeros(n),
            snow_psol=np.zeros(n),
            snow_pack=np.zeros(n),
            snow_thermal_state=np.zeros(n),
            snow_gratio=np.zeros(n),
            snow_pot_melt=np.zeros(n),
            snow_melt=np.zeros(n),
            snow_pliq_and_melt=np.zeros(n),
            snow_temp=np.zeros(n),
            snow_gthreshold=np.zeros(n),
            snow_glocalmax=np.zeros(n),
        )

        assert output.snow_pack is not None
        assert len(output.snow_pack) == n

    def test_to_dict_returns_all_fields(self) -> None:
        """Test to_dict contains all 12 fields."""
        n = 3
        output = SnowOutput(
            precip_raw=np.array([10.0, 20.0, 30.0]),
            snow_pliq=np.zeros(n),
            snow_psol=np.zeros(n),
            snow_pack=np.array([5.0, 10.0, 15.0]),
            snow_thermal_state=np.zeros(n),
            snow_gratio=np.zeros(n),
            snow_pot_melt=np.zeros(n),
            snow_melt=np.zeros(n),
            snow_pliq_and_melt=np.zeros(n),
            snow_temp=np.zeros(n),
            snow_gthreshold=np.zeros(n),
            snow_glocalmax=np.zeros(n),
        )

        result = output.to_dict()

        assert len(result) == 12
        assert "precip_raw" in result
        assert "snow_pack" in result
        assert "snow_melt" in result
        np.testing.assert_array_equal(result["precip_raw"], np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(result["snow_pack"], np.array([5.0, 10.0, 15.0]))

    def test_field_count_is_twelve(self) -> None:
        """Test there are exactly 12 snow fields."""
        field_count = len(fields(SnowOutput))

        assert field_count == 12


class TestModelOutput:
    """Tests for the ModelOutput frozen dataclass."""

    @pytest.fixture
    def sample_gr6j_output(self) -> GR6JOutput:
        """Provide a sample GR6JOutput for testing."""
        n = 5
        return GR6JOutput(
            pet=np.ones(n) * 3.0,
            precip=np.ones(n) * 10.0,
            production_store=np.ones(n) * 100.0,
            net_rainfall=np.ones(n) * 5.0,
            storage_infiltration=np.ones(n) * 2.0,
            actual_et=np.ones(n) * 2.5,
            percolation=np.ones(n) * 0.5,
            effective_rainfall=np.ones(n) * 4.5,
            q9=np.ones(n) * 4.0,
            q1=np.ones(n) * 0.5,
            routing_store=np.ones(n) * 50.0,
            exchange=np.ones(n) * 0.1,
            actual_exchange_routing=np.ones(n) * 0.05,
            actual_exchange_direct=np.ones(n) * 0.02,
            actual_exchange_total=np.ones(n) * 0.07,
            qr=np.ones(n) * 3.5,
            qrexp=np.ones(n) * 0.2,
            exponential_store=np.ones(n) * 10.0,
            qd=np.ones(n) * 0.3,
            streamflow=np.ones(n) * 4.0,
        )

    @pytest.fixture
    def sample_snow_output(self) -> SnowOutput:
        """Provide a sample SnowOutput for testing."""
        n = 5
        return SnowOutput(
            precip_raw=np.ones(n) * 15.0,
            snow_pliq=np.ones(n) * 10.0,
            snow_psol=np.ones(n) * 5.0,
            snow_pack=np.ones(n) * 20.0,
            snow_thermal_state=np.ones(n) * -2.0,
            snow_gratio=np.ones(n) * 0.8,
            snow_pot_melt=np.ones(n) * 3.0,
            snow_melt=np.ones(n) * 2.5,
            snow_pliq_and_melt=np.ones(n) * 12.5,
            snow_temp=np.ones(n) * -1.0,
            snow_gthreshold=np.ones(n) * 135.0,
            snow_glocalmax=np.ones(n) * 140.0,
        )

    @pytest.fixture
    def sample_time_array(self) -> np.ndarray:
        """Provide a sample time array for testing."""
        return pd.date_range("2020-01-01", periods=5, freq="D").to_numpy()

    def test_creates_with_gr6j_only(self, sample_gr6j_output: GR6JOutput, sample_time_array: np.ndarray) -> None:
        """Test creation without snow output."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
            snow=None,
        )

        assert output.snow is None
        assert output.gr6j is not None

    def test_creates_with_snow(
        self,
        sample_gr6j_output: GR6JOutput,
        sample_snow_output: SnowOutput,
        sample_time_array: np.ndarray,
    ) -> None:
        """Test creation with snow output."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
            snow=sample_snow_output,
        )

        assert output.snow is not None
        assert output.gr6j is not None

    def test_len_returns_timestep_count(self, sample_gr6j_output: GR6JOutput, sample_time_array: np.ndarray) -> None:
        """Test __len__ returns correct length."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
        )

        assert len(output) == 5

    def test_to_dataframe_without_snow(self, sample_gr6j_output: GR6JOutput, sample_time_array: np.ndarray) -> None:
        """Test to_dataframe produces DataFrame with 20 columns."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
        )

        df = output.to_dataframe()

        assert len(df.columns) == 20
        assert "streamflow" in df.columns
        assert df.index.name == "time"

    def test_to_dataframe_with_snow(
        self,
        sample_gr6j_output: GR6JOutput,
        sample_snow_output: SnowOutput,
        sample_time_array: np.ndarray,
    ) -> None:
        """Test to_dataframe produces DataFrame with 32 columns when snow enabled."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
            snow=sample_snow_output,
        )

        df = output.to_dataframe()

        assert len(df.columns) == 32  # 20 GR6J + 12 snow
        assert "streamflow" in df.columns
        assert "snow_pack" in df.columns

    def test_to_dataframe_index_is_time(self, sample_gr6j_output: GR6JOutput, sample_time_array: np.ndarray) -> None:
        """Test DataFrame index is the time array."""
        output = ModelOutput(
            time=sample_time_array,
            gr6j=sample_gr6j_output,
        )

        df = output.to_dataframe()

        assert df.index.name == "time"
        np.testing.assert_array_equal(df.index.values, sample_time_array)
        assert len(df) == 5
