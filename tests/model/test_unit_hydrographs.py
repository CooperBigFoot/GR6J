"""Tests for GR6J unit hydrograph functions.

Tests verify:
- Correct ordinate array lengths (UH1=20, UH2=40)
- Mass conservation (ordinates sum to 1)
- Non-negativity of all ordinates
- UH1 responds faster than UH2
- X4 parameter affects peak timing
- Convolution mechanics work correctly
"""

import numpy as np

from gr6j.model.unit_hydrographs import compute_uh_ordinates, convolve_uh


class TestComputeUhOrdinates:
    """Tests for compute_uh_ordinates function."""

    def test_uh1_has_correct_length(self) -> None:
        """UH1 ordinates should have length NH=20."""
        x4 = 2.0
        uh1_ordinates, _ = compute_uh_ordinates(x4)

        assert len(uh1_ordinates) == 20

    def test_uh2_has_correct_length(self) -> None:
        """UH2 ordinates should have length 2*NH=40."""
        x4 = 2.0
        _, uh2_ordinates = compute_uh_ordinates(x4)

        assert len(uh2_ordinates) == 40

    def test_uh1_ordinates_sum_to_one(self) -> None:
        """UH1 ordinates must sum to 1 for mass conservation."""
        test_x4_values = [0.5, 1.0, 2.0, 5.0, 10.0]

        for x4 in test_x4_values:
            uh1_ordinates, _ = compute_uh_ordinates(x4)
            assert np.isclose(uh1_ordinates.sum(), 1.0, atol=1e-10), (
                f"UH1 sum={uh1_ordinates.sum()} for x4={x4}, expected 1.0"
            )

    def test_uh2_ordinates_sum_to_one(self) -> None:
        """UH2 ordinates must sum to 1 for mass conservation."""
        test_x4_values = [0.5, 1.0, 2.0, 5.0, 10.0]

        for x4 in test_x4_values:
            _, uh2_ordinates = compute_uh_ordinates(x4)
            assert np.isclose(uh2_ordinates.sum(), 1.0, atol=1e-10), (
                f"UH2 sum={uh2_ordinates.sum()} for x4={x4}, expected 1.0"
            )

    def test_all_ordinates_non_negative(self) -> None:
        """All UH ordinates must be >= 0 (no negative probabilities)."""
        test_x4_values = [0.5, 1.0, 2.0, 5.0, 10.0]

        for x4 in test_x4_values:
            uh1_ordinates, uh2_ordinates = compute_uh_ordinates(x4)

            assert np.all(uh1_ordinates >= 0), f"UH1 has negative values for x4={x4}"
            assert np.all(uh2_ordinates >= 0), f"UH2 has negative values for x4={x4}"

    def test_uh1_response_faster_than_uh2(self) -> None:
        """Peak of UH1 should occur at or before peak of UH2.

        UH1 has base time X4, while UH2 has base time 2*X4, so UH1
        concentrates its mass earlier in time.
        """
        test_x4_values = [1.0, 2.0, 5.0]

        for x4 in test_x4_values:
            uh1_ordinates, uh2_ordinates = compute_uh_ordinates(x4)

            uh1_peak_index = np.argmax(uh1_ordinates)
            uh2_peak_index = np.argmax(uh2_ordinates)

            assert uh1_peak_index <= uh2_peak_index, (
                f"UH1 peak at {uh1_peak_index} > UH2 peak at {uh2_peak_index} for x4={x4}"
            )

    def test_varying_x4_changes_peak_timing(self) -> None:
        """Larger X4 values should result in later peak timing.

        The X4 parameter controls the time base of the unit hydrographs,
        so increasing X4 should shift the peak to later indices.
        """
        x4_small = 1.0
        x4_large = 5.0

        uh1_small, uh2_small = compute_uh_ordinates(x4_small)
        uh1_large, uh2_large = compute_uh_ordinates(x4_large)

        # UH1 peak timing
        uh1_peak_small = np.argmax(uh1_small)
        uh1_peak_large = np.argmax(uh1_large)

        assert uh1_peak_large >= uh1_peak_small, (
            f"UH1 peak should shift later with larger X4: {uh1_peak_small} vs {uh1_peak_large}"
        )

        # UH2 peak timing
        uh2_peak_small = np.argmax(uh2_small)
        uh2_peak_large = np.argmax(uh2_large)

        assert uh2_peak_large >= uh2_peak_small, (
            f"UH2 peak should shift later with larger X4: {uh2_peak_small} vs {uh2_peak_large}"
        )


class TestConvolveUh:
    """Tests for convolve_uh function."""

    def test_output_is_first_state_before_update(self) -> None:
        """The output should be uh_states[0] before any update."""
        x4 = 2.0
        uh1_ordinates, _ = compute_uh_ordinates(x4)

        # Initialize states with known values
        uh_states = np.arange(20, dtype=np.float64)  # [0, 1, 2, ..., 19]
        pr_input = 10.0

        _, output = convolve_uh(uh_states, pr_input, uh1_ordinates)

        assert output == 0.0, f"Output should be initial uh_states[0]=0, got {output}"

        # Test with non-zero first element
        uh_states[0] = 5.5
        _, output = convolve_uh(uh_states, pr_input, uh1_ordinates)

        assert output == 5.5, f"Output should be 5.5, got {output}"

    def test_states_shift_and_add_input(self) -> None:
        """Verify convolution: new_states[k] = uh_states[k+1] + ordinates[k] * pr_input."""
        x4 = 2.0
        uh1_ordinates, _ = compute_uh_ordinates(x4)

        # Initialize with known values
        uh_states = np.ones(20, dtype=np.float64) * 2.0  # All 2.0
        pr_input = 1.0

        new_states, _ = convolve_uh(uh_states, pr_input, uh1_ordinates)

        # For k = 0 to 18: new_states[k] = uh_states[k+1] + uh1_ordinates[k] * pr_input
        for k in range(19):
            expected = uh_states[k + 1] + uh1_ordinates[k] * pr_input
            assert np.isclose(new_states[k], expected), f"new_states[{k}]={new_states[k]}, expected {expected}"

        # Last element: new_states[-1] = uh1_ordinates[-1] * pr_input
        expected_last = uh1_ordinates[-1] * pr_input
        assert np.isclose(new_states[-1], expected_last), f"new_states[-1]={new_states[-1]}, expected {expected_last}"

    def test_zero_input_shifts_states(self) -> None:
        """With pr_input=0, states should just shift without adding anything."""
        x4 = 2.0
        uh1_ordinates, _ = compute_uh_ordinates(x4)

        # Initialize with increasing values
        uh_states = np.arange(20, dtype=np.float64)  # [0, 1, 2, ..., 19]
        pr_input = 0.0

        new_states, output = convolve_uh(uh_states, pr_input, uh1_ordinates)

        # Output is uh_states[0]
        assert output == 0.0

        # new_states[k] = uh_states[k+1] (pure shift)
        for k in range(19):
            assert new_states[k] == uh_states[k + 1], f"new_states[{k}]={new_states[k]}, expected {uh_states[k + 1]}"

        # Last element should be 0 (no input contribution)
        assert new_states[-1] == 0.0, f"new_states[-1]={new_states[-1]}, expected 0.0"

    def test_impulse_response_matches_ordinates(self) -> None:
        """A single unit impulse should produce outputs matching the ordinates.

        If we input a unit impulse (1.0) at time 0 with zero initial states,
        the successive outputs should match the UH ordinates.
        """
        x4 = 2.0
        uh1_ordinates, _ = compute_uh_ordinates(x4)

        # Start with zero states
        uh_states = np.zeros(20, dtype=np.float64)

        # Apply unit impulse
        pr_input = 1.0
        outputs = []

        # First convolution: input the impulse
        uh_states, output = convolve_uh(uh_states, pr_input, uh1_ordinates)
        outputs.append(output)  # Should be 0 (initial state)

        # Subsequent convolutions with zero input to observe the impulse response
        for _ in range(len(uh1_ordinates)):
            uh_states, output = convolve_uh(uh_states, 0.0, uh1_ordinates)
            outputs.append(output)

        # The outputs after the initial zero should match the ordinates
        # outputs[0] = 0 (initial state)
        # outputs[1] = uh1_ordinates[0]
        # outputs[2] = uh1_ordinates[1]
        # etc.
        for i, ordinate in enumerate(uh1_ordinates):
            assert np.isclose(outputs[i + 1], ordinate, atol=1e-12), (
                f"Output at t={i + 1} is {outputs[i + 1]}, expected ordinate {ordinate}"
            )
