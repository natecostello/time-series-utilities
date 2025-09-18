"""
Test cases for trigger detection and pre-event removal functionality.

Following TDD methodology with comprehensive test coverage.
"""

import unittest

import numpy as np
import pandas as pd

from src.series_utilities.core.trigger_detection import (
    detect_event_trigger,
    remove_pre_event_data,
)


class TestTriggerDetection(unittest.TestCase):
    """Test cases for event trigger detection functionality."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create synthetic test data: baseline noise + shock event
        np.random.seed(42)  # For reproducible tests

        # Baseline noise (50 points)
        self.baseline_length = 50
        self.baseline_noise = np.random.normal(0, 0.1, self.baseline_length)

        # Shock event (30 points)
        self.event_length = 30
        shock_amplitude = 5.0
        self.shock_event = shock_amplitude * np.exp(
            -np.linspace(0, 3, self.event_length)
        )

        # Combined signal
        self.time_data = np.arange(self.baseline_length + self.event_length) * 0.01
        self.acceleration_data = np.concatenate([self.baseline_noise, self.shock_event])

        # Expected trigger index (start of shock event)
        self.expected_trigger_index = self.baseline_length

    def test_detect_event_trigger_scientific_parameter_names(self) -> None:
        """Test trigger detection with scientific parameter names."""
        # This test verifies the new API with independent_data and dependent_data parameters
        trigger_index = detect_event_trigger(
            independent_data=self.time_data, 
            dependent_data=self.acceleration_data
        )
        
        # Should detect trigger near the expected location (within tolerance)
        tolerance = 5  # Allow some variation
        self.assertAlmostEqual(
            trigger_index, self.expected_trigger_index, delta=tolerance,
            msg="Trigger should be detected near expected location"
        )
        
    def test_detect_event_trigger_basic_functionality(self) -> None:
        """Test basic trigger detection with default parameters."""
        trigger_index = detect_event_trigger(self.time_data, self.acceleration_data)

        # Should detect trigger near the expected location (within tolerance)
        tolerance = 5  # Allow some variation
        self.assertAlmostEqual(
            trigger_index,
            self.expected_trigger_index,
            delta=tolerance,
            msg="Trigger detection should identify event onset accurately",
        )

    def test_detect_event_trigger_with_threshold_parameter(self) -> None:
        """Test trigger detection with custom threshold."""
        # Test with conservative threshold (higher sigma multiplier)
        conservative_trigger = detect_event_trigger(
            self.time_data, self.acceleration_data, threshold_sigma=6.0
        )

        # Test with sensitive threshold (lower sigma multiplier)
        sensitive_trigger = detect_event_trigger(
            self.time_data, self.acceleration_data, threshold_sigma=2.0
        )

        # Conservative should be later (more selective)
        # Sensitive should be earlier (less selective)
        self.assertGreaterEqual(
            conservative_trigger,
            sensitive_trigger,
            msg="Conservative threshold should trigger later than sensitive",
        )

    def test_detect_event_trigger_bidirectional_detection(self) -> None:
        """Test detection of both positive and negative events."""
        # Test positive shock (already tested above)
        pos_trigger = detect_event_trigger(self.time_data, self.acceleration_data)

        # Test negative shock
        negative_acceleration = self.acceleration_data.copy()
        negative_acceleration[self.baseline_length :] *= -1

        neg_trigger = detect_event_trigger(self.time_data, negative_acceleration)

        # Both should detect at similar positions
        tolerance = 5
        self.assertAlmostEqual(
            pos_trigger,
            neg_trigger,
            delta=tolerance,
            msg="Should detect both positive and negative events",
        )

    def test_detect_event_trigger_no_event(self) -> None:
        """Test behavior when no significant event is present."""
        # Pure noise signal
        pure_noise = np.random.normal(0, 0.1, 100)
        time_pure = np.arange(100) * 0.01

        trigger_index = detect_event_trigger(time_pure, pure_noise)

        # Should return None or -1 when no event detected
        self.assertIsNone(
            trigger_index, msg="Should return None when no significant event detected"
        )

    def test_detect_event_trigger_input_validation(self) -> None:
        """Test input validation and error handling."""
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            detect_event_trigger(
                self.time_data[:-10], self.acceleration_data  # Shorter time array
            )

        # Test invalid threshold
        with self.assertRaises(ValueError):
            detect_event_trigger(
                self.time_data,
                self.acceleration_data,
                threshold_sigma=-1.0,  # Negative threshold
            )

    def test_remove_pre_event_data_basic_functionality(self) -> None:
        """Test basic pre-event data removal."""
        trigger_index = self.expected_trigger_index

        filtered_time, filtered_data = remove_pre_event_data(
            self.time_data, self.acceleration_data, trigger_index
        )

        # Check that pre-event data is removed
        expected_length = len(self.acceleration_data) - trigger_index
        self.assertEqual(
            len(filtered_time),
            expected_length,
            msg="Filtered time array should have correct length",
        )
        self.assertEqual(
            len(filtered_data),
            expected_length,
            msg="Filtered data array should have correct length",
        )

        # Check that remaining data starts from trigger point
        np.testing.assert_array_almost_equal(
            filtered_data,
            self.acceleration_data[trigger_index:],
            err_msg="Filtered data should match original data from trigger point",
        )

    def test_remove_pre_event_data_with_padding(self) -> None:
        """Test pre-event removal with padding (keep some baseline)."""
        trigger_index = self.expected_trigger_index
        padding_samples = 10

        filtered_time, filtered_data = remove_pre_event_data(
            self.time_data,
            self.acceleration_data,
            trigger_index,
            padding_samples=padding_samples,
        )

        # Should keep padding samples before trigger
        expected_start_index = max(0, trigger_index - padding_samples)
        expected_length = len(self.acceleration_data) - expected_start_index

        self.assertEqual(
            len(filtered_data),
            expected_length,
            msg="Should include padding samples before trigger",
        )

    def test_remove_pre_event_data_independent_axis_normalization(self) -> None:
        """Test independent axis normalization after pre-event removal."""
        trigger_index = self.expected_trigger_index

        # Test with normalization enabled (default)
        filtered_time, filtered_data = remove_pre_event_data(
            self.time_data,
            self.acceleration_data,
            trigger_index,
            normalize_independent=True,
        )

        # Independent axis should start at zero
        self.assertAlmostEqual(
            filtered_time[0],
            0.0,
            places=10,
            msg="Independent axis should start at zero when normalized",
        )

        # Check that relative spacing is preserved
        original_spacing = (
            self.time_data[trigger_index + 1] - self.time_data[trigger_index]
        )
        filtered_spacing = filtered_time[1] - filtered_time[0]
        self.assertAlmostEqual(
            filtered_spacing,
            original_spacing,
            places=10,
            msg="Time spacing should be preserved after normalization",
        )

        # Test with normalization disabled
        filtered_time_no_norm, filtered_data_no_norm = remove_pre_event_data(
            self.time_data,
            self.acceleration_data,
            trigger_index,
            normalize_independent=False,
        )

        # Should preserve original time values
        self.assertAlmostEqual(
            filtered_time_no_norm[0],
            self.time_data[trigger_index],
            places=10,
            msg="Should preserve original time values when normalization disabled",
        )

        # Data should be identical regardless of normalization
        np.testing.assert_array_equal(
            filtered_data,
            filtered_data_no_norm,
            err_msg="Dependent data should be identical regardless of time normalization",
        )

    def test_detect_event_trigger_intelligent_baseline_backup(self) -> None:
        """Test intelligent baseline backup in detect_event_trigger."""
        # Create a test case where backup will make a difference
        time_data = np.linspace(0, 2, 200)
        baseline_level = 1.0
        noise_level = 0.1
        signal_data = baseline_level + noise_level * np.random.random(200)

        # Add gradual onset that will trigger backup mechanism
        onset_idx = 80
        ramp_length = 40
        for i in range(onset_idx, min(onset_idx + ramp_length, len(signal_data))):
            progress = (i - onset_idx) / ramp_length
            signal_data[i] += 1.0 * (progress**2)  # Gradual quadratic rise

        # Add main event after ramp
        main_event_idx = onset_idx + ramp_length
        if main_event_idx < len(signal_data):
            signal_data[main_event_idx:] += 2.0

        # Test with different backup_sigma values
        conservative_backup = detect_event_trigger(
            time_data,
            signal_data,
            threshold_sigma=4.0,
            backup_sigma=3.0,  # Conservative backup
        )

        sensitive_backup = detect_event_trigger(
            time_data,
            signal_data,
            threshold_sigma=4.0,
            backup_sigma=1.5,  # More sensitive backup
        )

        # Both should detect triggers, but sensitive backup should be earlier
        self.assertIsNotNone(conservative_backup)
        self.assertIsNotNone(sensitive_backup)

        # Sensitive backup should find earlier onset (or same at worst)
        self.assertLessEqual(
            sensitive_backup,
            conservative_backup,
            "Sensitive backup should detect onset earlier or same",
        )

        # Test remove_pre_event_data with intelligent trigger (no backup_trigger param needed)
        filtered_time, filtered_data = remove_pre_event_data(
            time_data, signal_data, sensitive_backup
        )

        # Verify the intelligent trigger detection provides good results
        self.assertIsNotNone(filtered_time)
        self.assertIsNotNone(filtered_data)
        self.assertGreater(
            len(filtered_data),
            0,
            "Should have data after intelligent trigger detection",
        )

    def test_remove_pre_event_data_edge_cases(self) -> None:
        """Test edge cases for pre-event removal."""
        # Test trigger at beginning
        filtered_time, filtered_data = remove_pre_event_data(
            self.time_data, self.acceleration_data, 0  # Trigger at start
        )

        # Should return original data
        np.testing.assert_array_equal(
            filtered_data,
            self.acceleration_data,
            err_msg="No removal should occur if trigger is at start",
        )

        # Test trigger beyond data length
        with self.assertRaises(ValueError):
            remove_pre_event_data(
                self.time_data, self.acceleration_data, len(self.acceleration_data) + 10
            )


class TestTriggerDetectionWithPandas(unittest.TestCase):
    """Test cases for trigger detection with pandas DataFrame input."""

    def setUp(self) -> None:
        """Set up pandas DataFrame test fixtures."""
        # Create test data similar to numpy version
        np.random.seed(42)
        baseline_length = 50
        event_length = 30

        time_data = np.arange(baseline_length + event_length) * 0.01
        baseline_noise = np.random.normal(0, 0.1, baseline_length)
        shock_event = 5.0 * np.exp(-np.linspace(0, 3, event_length))
        acceleration_data = np.concatenate([baseline_noise, shock_event])

        # Create DataFrame
        self.df = pd.DataFrame({"time": time_data, "acceleration": acceleration_data})

        self.expected_trigger_index = baseline_length

    def test_detect_event_trigger_with_dataframe(self) -> None:
        """Test trigger detection with DataFrame input."""
        trigger_index = detect_event_trigger(self.df["time"], self.df["acceleration"])

        tolerance = 5
        self.assertAlmostEqual(
            trigger_index,
            self.expected_trigger_index,
            delta=tolerance,
            msg="Should work with pandas Series input",
        )

    def test_remove_pre_event_data_with_dataframe(self) -> None:
        """Test pre-event removal returning DataFrame."""
        trigger_index = self.expected_trigger_index

        # This should return a filtered DataFrame
        filtered_df = remove_pre_event_data(self.df, trigger_index=trigger_index)

        expected_length = len(self.df) - trigger_index
        self.assertEqual(
            len(filtered_df),
            expected_length,
            msg="Filtered DataFrame should have correct length",
        )

        # Check that it's a DataFrame
        self.assertIsInstance(
            filtered_df,
            pd.DataFrame,
            msg="Should return DataFrame when DataFrame input provided",
        )

    def test_remove_pre_event_data_dataframe_normalization(self) -> None:
        """Test DataFrame with independent axis normalization."""
        trigger_index = self.expected_trigger_index

        # Test with normalization (default)
        filtered_df_norm = remove_pre_event_data(
            self.df, trigger_index=trigger_index, normalize_independent=True
        )

        # Test without normalization
        filtered_df_no_norm = remove_pre_event_data(
            self.df, trigger_index=trigger_index, normalize_independent=False
        )

        # Verify normalization resets first column (time) to start at zero
        first_column = filtered_df_norm.columns[0]
        self.assertAlmostEqual(
            filtered_df_norm[first_column].iloc[0],
            0.0,
            places=10,
            msg="Normalized DataFrame should start at zero for independent axis",
        )

        # Verify without normalization preserves original values
        original_first_time = self.df[first_column].iloc[trigger_index]
        self.assertAlmostEqual(
            filtered_df_no_norm[first_column].iloc[0],
            original_first_time,
            places=10,
            msg="Non-normalized DataFrame should preserve original time values",
        )

    def test_remove_pre_event_data_dataframe_intelligent_trigger(self) -> None:
        """Test DataFrame with intelligent trigger detection (no backup_trigger param)."""
        # Extract arrays from DataFrame for trigger detection
        time_array = self.df[self.df.columns[0]].values
        data_array = self.df[self.df.columns[1]].values

        # Use intelligent trigger detection
        intelligent_trigger_index = detect_event_trigger(
            time_array, data_array, threshold_sigma=3.0, backup_sigma=2.0
        )

        # Test DataFrame filtering with intelligent trigger
        filtered_df = remove_pre_event_data(
            self.df, trigger_index=intelligent_trigger_index
        )

        # Verify results
        self.assertIsNotNone(intelligent_trigger_index)
        self.assertGreater(len(filtered_df), 0, "Should have data after filtering")
        self.assertEqual(
            len(filtered_df.columns),
            len(self.df.columns),
            "Should preserve all columns",
        )

    def test_remove_pre_event_data_dataframe_with_normalization(self) -> None:
        """Test DataFrame with normalization (main feature preserved)."""
        trigger_index = self.expected_trigger_index

        filtered_df = remove_pre_event_data(
            self.df, trigger_index=trigger_index, normalize_independent=True
        )

        # Should start at time=0 due to normalization
        time_column = filtered_df.columns[0]
        self.assertAlmostEqual(
            filtered_df[time_column].iloc[0],
            0.0,
            places=10,
            msg="Normalization: time should start at zero",
        )

        # Should have expected length
        expected_length = len(self.df) - trigger_index
        self.assertEqual(
            len(filtered_df), expected_length, msg="Should have correct filtered length"
        )


if __name__ == "__main__":
    unittest.main()
