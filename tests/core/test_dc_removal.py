"""
Test cases for DC component removal through integral mean subtraction.

Tests the mathematical operation:
corrected_signal(t) = original_signal(t) - (1/T) ∫ original_signal(t) dt
Following TDD methodology with comprehensive test coverage.
"""

import unittest

import numpy as np
import pandas as pd

from src.series_utilities.core.dc_removal import remove_dc_component


class TestDCComponentRemoval(unittest.TestCase):
    """Test cases for DC component removal functionality."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create test signals with known DC components
        self.time_uniform = np.linspace(0, 10, 101)  # Uniform time spacing
        self.dt = self.time_uniform[1] - self.time_uniform[0]

        # Test case 1: Pure DC offset
        self.dc_offset = 5.0
        self.pure_dc_signal = np.full_like(self.time_uniform, self.dc_offset)

        # Test case 2: Sinusoidal signal with DC offset
        self.freq = 1.0  # Hz
        self.amplitude = 2.0
        self.sine_with_dc = (
            self.amplitude * np.sin(2 * np.pi * self.freq * self.time_uniform)
            + self.dc_offset
        )

        # Test case 3: Acceleration-like signal (impulse response with DC drift)
        # Simulates shock event that should have zero velocity change
        self.shock_signal = np.exp(-self.time_uniform) * np.cos(5 * self.time_uniform)
        # Add artificial DC component that needs removal
        self.shock_with_dc = self.shock_signal + 0.2  # Small DC offset

        # Test case 4: Non-uniform time spacing
        self.time_nonuniform = np.array(
            [0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5, 5.5]
        )
        self.signal_nonuniform = np.sin(self.time_nonuniform) + 1.5

    def test_remove_dc_component_scientific_parameter_names(self) -> None:
        """Test DC removal with scientific parameter names."""
        # This test verifies the new API with independent_data and dependent_data parameters
        corrected_time, corrected_signal = remove_dc_component(
            independent_data=self.time_uniform, 
            dependent_data=self.pure_dc_signal
        )

        # Time should remain unchanged
        np.testing.assert_array_equal(
            corrected_time,
            self.time_uniform,
            err_msg="Time data should remain unchanged",
        )

        # Signal should be zero after DC removal (within numerical precision)
        np.testing.assert_array_almost_equal(
            corrected_signal,
            np.zeros_like(self.pure_dc_signal),
            decimal=10,
            err_msg="Pure DC signal should become zero after correction",
        )

    def test_remove_dc_component_pure_dc_signal(self) -> None:
        """Test DC removal on pure DC signal - should result in zero."""
        corrected_time, corrected_signal = remove_dc_component(
            self.time_uniform, self.pure_dc_signal
        )

        # Time should remain unchanged
        np.testing.assert_array_equal(
            corrected_time,
            self.time_uniform,
            err_msg="Time data should remain unchanged",
        )

        # Signal should be approximately zero everywhere
        np.testing.assert_array_almost_equal(
            corrected_signal,
            np.zeros_like(corrected_signal),
            decimal=10,
            err_msg="Pure DC signal should become zero after DC removal",
        )

    def test_remove_dc_component_sine_wave_with_dc(self) -> None:
        """Test DC removal on sine wave with DC offset - should preserve AC, remove DC."""
        corrected_time, corrected_signal = remove_dc_component(
            self.time_uniform, self.sine_with_dc
        )

        # The corrected signal should be the pure sine wave (DC removed)
        expected_signal = self.amplitude * np.sin(
            2 * np.pi * self.freq * self.time_uniform
        )

        np.testing.assert_array_almost_equal(
            corrected_signal,
            expected_signal,
            decimal=10,
            err_msg="DC removal should preserve sine wave shape, remove offset",
        )

    def test_remove_dc_component_zero_net_change_verification(self) -> None:
        """Test that DC removal enforces zero net change (integral = 0)."""
        corrected_time, corrected_signal = remove_dc_component(
            self.time_uniform, self.shock_with_dc
        )

        # Calculate integral using trapezoidal rule
        integral_corrected = np.trapezoid(corrected_signal, corrected_time)

        # Should be approximately zero (within numerical precision)
        self.assertAlmostEqual(
            integral_corrected,
            0.0,
            places=10,
            msg="Integral of corrected signal should be zero (net change = 0)",
        )

    def test_remove_dc_component_mathematical_formula_verification(self) -> None:
        """Test that the implementation matches the mathematical formula."""
        # Test with shock signal
        corrected_time, corrected_signal = remove_dc_component(
            self.time_uniform, self.shock_with_dc
        )

        # Calculate the integral mean manually
        # (1/T) ∫ original_signal(t) dt where T is total time duration
        T = self.time_uniform[-1] - self.time_uniform[0]
        integral_original = np.trapezoid(self.shock_with_dc, self.time_uniform)
        integral_mean = integral_original / T

        # Expected corrected signal
        expected_corrected = self.shock_with_dc - integral_mean

        np.testing.assert_array_almost_equal(
            corrected_signal,
            expected_corrected,
            decimal=10,
            err_msg="Implementation should match mathematical formula",
        )

    def test_remove_dc_component_nonuniform_time_spacing(self) -> None:
        """Test DC removal with non-uniform time spacing."""
        corrected_time, corrected_signal = remove_dc_component(
            self.time_nonuniform, self.signal_nonuniform
        )

        # Verify zero net change even with irregular spacing
        integral_corrected = np.trapezoid(corrected_signal, corrected_time)

        self.assertAlmostEqual(
            integral_corrected,
            0.0,
            places=8,  # Slightly less precision for non-uniform spacing
            msg="DC removal should work with non-uniform time spacing",
        )

    def test_remove_dc_component_input_validation(self) -> None:
        """Test input validation and error handling."""
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            remove_dc_component(
                self.time_uniform[:-5], self.pure_dc_signal  # Shorter time array
            )

        # Test empty arrays
        with self.assertRaises(ValueError):
            remove_dc_component(np.array([]), np.array([]))

    def test_remove_dc_component_single_point(self) -> None:
        """Test edge case with single data point."""
        single_time = np.array([1.0])
        single_signal = np.array([5.0])

        corrected_time, corrected_signal = remove_dc_component(
            single_time, single_signal
        )

        # Single point should become zero after DC removal
        self.assertEqual(corrected_signal[0], 0.0)

    def test_remove_dc_component_preserves_zero_mean_signals(self) -> None:
        """Test that signals with zero mean are unchanged."""
        # Create zero-mean signal
        zero_mean_signal = np.sin(2 * np.pi * self.time_uniform)

        corrected_time, corrected_signal = remove_dc_component(
            self.time_uniform, zero_mean_signal
        )

        # Should remain essentially unchanged
        np.testing.assert_array_almost_equal(
            corrected_signal,
            zero_mean_signal,
            decimal=10,
            err_msg="Zero-mean signals should be preserved",
        )

    def test_remove_dc_component_acceleration_velocity_constraint(self) -> None:
        """Test the specific use case: acceleration signal with ΔV = 0 constraint."""
        # Create acceleration signal that should result in zero velocity change
        # but has a DC component that would cause drift

        # Damped oscillation (physically realistic acceleration)
        damping = 0.5
        frequency = 2.0
        acceleration = np.exp(-damping * self.time_uniform) * np.sin(
            2 * np.pi * frequency * self.time_uniform
        )

        # Add DC component that would cause velocity drift
        acceleration_with_drift = acceleration + 0.1

        # Remove DC component
        corrected_time, corrected_acceleration = remove_dc_component(
            self.time_uniform, acceleration_with_drift
        )

        # Integrate corrected acceleration to get velocity
        velocity = np.trapezoid(corrected_acceleration, corrected_time)

        # Velocity change should be approximately zero
        self.assertAlmostEqual(
            velocity,
            0.0,
            places=10,
            msg="DC removal should enforce ΔV = 0 for acceleration signals",
        )


class TestDCComponentRemovalWithPandas(unittest.TestCase):
    """Test cases for DC component removal with pandas DataFrame input."""

    def setUp(self) -> None:
        """Set up pandas DataFrame test fixtures."""
        time_data = np.linspace(0, 5, 51)
        signal_data = np.sin(2 * np.pi * time_data) + 2.5  # Sine with DC offset

        self.df = pd.DataFrame(
            {
                "time": time_data,
                "acceleration": signal_data,
                "other_column": np.random.randn(len(time_data)),
            }
        )

    def test_remove_dc_component_with_pandas_series(self) -> None:
        """Test DC removal with pandas Series input."""
        corrected_time, corrected_signal = remove_dc_component(
            self.df["time"], self.df["acceleration"]
        )

        # Should work with Series input
        self.assertIsInstance(corrected_time, np.ndarray)
        self.assertIsInstance(corrected_signal, np.ndarray)

        # Verify zero integral
        integral = np.trapezoid(corrected_signal, corrected_time)
        self.assertAlmostEqual(integral, 0.0, places=10)

    def test_remove_dc_component_dataframe_integration(self) -> None:
        """Test integration with DataFrame workflow."""
        # Extract time and signal
        time_data = self.df["time"]
        signal_data = self.df["acceleration"]

        # Apply DC removal
        corrected_time, corrected_signal = remove_dc_component(time_data, signal_data)

        # Create new DataFrame with corrected data
        corrected_df = self.df.copy()
        corrected_df["acceleration_corrected"] = corrected_signal

        # Verify the correction worked
        integral_original = np.trapezoid(self.df["acceleration"], self.df["time"])
        integral_corrected = np.trapezoid(
            corrected_df["acceleration_corrected"], corrected_df["time"]
        )

        # Original should have non-zero integral, corrected should be zero
        self.assertNotAlmostEqual(integral_original, 0.0, places=5)
        self.assertAlmostEqual(integral_corrected, 0.0, places=10)


class TestDCComponentRemovalNumericalStability(unittest.TestCase):
    """Test numerical stability and precision of DC component removal."""

    def test_remove_dc_component_numerical_precision(self) -> None:
        """Test numerical precision with various signal magnitudes."""
        time_data = np.linspace(0, 1, 1001)  # High resolution

        # Test with different signal magnitudes
        magnitudes = [1e-6, 1e-3, 1.0, 1e3, 1e6]

        for mag in magnitudes:
            with self.subTest(magnitude=mag):
                # Create signal with known DC component
                signal = mag * (np.sin(10 * np.pi * time_data) + 0.1)  # 10% DC offset

                corrected_time, corrected_signal = remove_dc_component(
                    time_data, signal
                )

                # Check zero integral regardless of magnitude
                integral = np.trapezoid(corrected_signal, corrected_time)
                relative_error = abs(integral) / (mag * 0.1 if mag > 0 else 1e-12)

                self.assertLess(
                    relative_error,
                    1e-10,
                    msg=f"Numerical precision should be maintained at magnitude {mag}",
                )

    def test_remove_dc_component_extreme_dc_values(self) -> None:
        """Test with extreme DC values."""
        time_data = np.linspace(0, 2, 201)

        extreme_dc_values = [-1e6, -100, 0, 100, 1e6]

        for dc_value in extreme_dc_values:
            with self.subTest(dc_value=dc_value):
                # Signal is pure sine + extreme DC
                signal = np.sin(np.pi * time_data) + dc_value

                corrected_time, corrected_signal = remove_dc_component(
                    time_data, signal
                )

                # Should still enforce zero integral
                integral = np.trapezoid(corrected_signal, corrected_time)
                self.assertAlmostEqual(
                    integral,
                    0.0,
                    places=8,
                    msg=f"Should handle extreme DC value {dc_value}",
                )


if __name__ == "__main__":
    unittest.main()
