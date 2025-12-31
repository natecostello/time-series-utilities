"""
Test cases for input validation and flexible input support.

Tests support for NumPy arrays, Pandas Series, and Pandas DataFrames
with dependent/independent column pairs.
"""

import unittest

import numpy as np
import pandas as pd

from src.series_utilities.utils.input_validation import (
    extract_dataframe_columns,
    normalize_input_arrays,
    validate_input_data,
)


class TestInputValidation(unittest.TestCase):
    """Test cases for basic input validation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.time_array = np.linspace(0, 10, 101)
        self.signal_array = np.sin(self.time_array)

        self.time_series = pd.Series(self.time_array, name="time")
        self.signal_series = pd.Series(self.signal_array, name="signal")

        self.df = pd.DataFrame(
            {
                "time": self.time_array,
                "acceleration": self.signal_array,
                "velocity": np.cumsum(self.signal_array) * 0.1,
                "position": np.random.randn(len(self.time_array)),
            }
        )

    def test_validate_input_data_scientific_parameter_names(self) -> None:
        """Test validation with scientific parameter names."""
        # This test verifies the new API with independent_data and dependent_data parameters
        result = validate_input_data(
            independent_data=self.time_array, 
            dependent_data=self.signal_array
        )
        self.assertTrue(result, "Should validate matching arrays with scientific names")

    def test_validate_input_data_valid_numpy_arrays(self) -> None:
        """Test validation with valid NumPy arrays."""
        result = validate_input_data(self.time_array, self.signal_array)
        self.assertTrue(result, "Should validate matching NumPy arrays")

    def test_validate_input_data_valid_pandas_series(self) -> None:
        """Test validation with valid Pandas Series."""
        result = validate_input_data(self.time_series, self.signal_series)
        self.assertTrue(result, "Should validate matching Pandas Series")

    def test_validate_input_data_mixed_types(self) -> None:
        """Test validation with mixed array types."""
        result = validate_input_data(self.time_array, self.signal_series)
        self.assertTrue(result, "Should validate mixed NumPy array and Pandas Series")

    def test_validate_input_data_mismatched_lengths(self) -> None:
        """Test validation fails with mismatched lengths."""
        short_array = self.signal_array[:-10]

        with self.assertRaises(
            ValueError, msg="Should raise error for mismatched lengths"
        ):
            validate_input_data(self.time_array, short_array)

    def test_validate_input_data_empty_arrays(self) -> None:
        """Test validation fails with empty arrays."""
        empty_array = np.array([])

        with self.assertRaises(ValueError, msg="Should raise error for empty arrays"):
            validate_input_data(empty_array, empty_array)

    def test_validate_input_data_single_element(self) -> None:
        """Test validation with single element arrays."""
        single_time = np.array([1.0])
        single_signal = np.array([2.5])

        result = validate_input_data(single_time, single_signal)
        self.assertTrue(result, "Should validate single element arrays")


class TestInputNormalization(unittest.TestCase):
    """Test cases for input normalization functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.time_list = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.signal_list = [0.0, 0.5, 1.0, 0.5, 0.0]
        self.time_array = np.array(self.time_list)
        self.signal_array = np.array(self.signal_list)
        self.time_series = pd.Series(self.time_array)
        self.signal_series = pd.Series(self.signal_array)

    def test_normalize_input_arrays_scientific_parameter_names(self) -> None:
        """Test normalization with scientific parameter names."""
        # This test verifies the new API with independent_data and dependent_data parameters
        time_norm, signal_norm = normalize_input_arrays(
            independent_data=self.time_array, 
            dependent_data=self.signal_array
        )

        self.assertIsInstance(time_norm, np.ndarray, "Should return NumPy arrays")
        self.assertIsInstance(signal_norm, np.ndarray, "Should return NumPy arrays")
        np.testing.assert_array_equal(time_norm, self.time_array)
        np.testing.assert_array_equal(signal_norm, self.signal_array)

    def test_normalize_input_arrays_scientific_return_values(self) -> None:
        """Test that the function returns scientifically named arrays."""
        # This test verifies the return values have scientific naming
        independent_norm, dependent_norm = normalize_input_arrays(
            independent_data=self.time_array, 
            dependent_data=self.signal_array
        )

        # The returned arrays should be scientifically named (per docstring)
        self.assertIsInstance(independent_norm, np.ndarray, "Should return independent array")
        self.assertIsInstance(dependent_norm, np.ndarray, "Should return dependent array")
        np.testing.assert_array_equal(independent_norm, self.time_array)
        np.testing.assert_array_equal(dependent_norm, self.signal_array)

    def test_normalize_input_arrays_numpy_arrays(self) -> None:
        """Test normalization of NumPy arrays."""
        time_norm, signal_norm = normalize_input_arrays(
            self.time_array, self.signal_array
        )

        self.assertIsInstance(time_norm, np.ndarray, "Should return NumPy arrays")
        self.assertIsInstance(signal_norm, np.ndarray, "Should return NumPy arrays")
        np.testing.assert_array_equal(time_norm, self.time_array)
        np.testing.assert_array_equal(signal_norm, self.signal_array)

    def test_normalize_input_arrays_pandas_series(self) -> None:
        """Test normalization of Pandas Series."""
        time_norm, signal_norm = normalize_input_arrays(
            self.time_series, self.signal_series
        )

        self.assertIsInstance(time_norm, np.ndarray, "Should convert to NumPy arrays")
        self.assertIsInstance(signal_norm, np.ndarray, "Should convert to NumPy arrays")
        np.testing.assert_array_equal(time_norm, self.time_array)
        np.testing.assert_array_equal(signal_norm, self.signal_array)

    def test_normalize_input_arrays_python_lists(self) -> None:
        """Test normalization of Python lists."""
        time_norm, signal_norm = normalize_input_arrays(
            self.time_list, self.signal_list
        )

        self.assertIsInstance(time_norm, np.ndarray, "Should convert to NumPy arrays")
        self.assertIsInstance(signal_norm, np.ndarray, "Should convert to NumPy arrays")
        np.testing.assert_array_equal(time_norm, self.time_array)
        np.testing.assert_array_equal(signal_norm, self.signal_array)

    def test_normalize_input_arrays_mixed_types(self) -> None:
        """Test normalization of mixed input types."""
        time_norm, signal_norm = normalize_input_arrays(
            self.time_list, self.signal_series
        )

        self.assertIsInstance(time_norm, np.ndarray, "Should convert to NumPy arrays")
        self.assertIsInstance(signal_norm, np.ndarray, "Should convert to NumPy arrays")
        np.testing.assert_array_equal(time_norm, self.time_array)
        np.testing.assert_array_equal(signal_norm, self.signal_array)

    def test_normalize_input_arrays_preserves_data_types(self) -> None:
        """Test that normalization preserves appropriate data types."""
        # Integer input
        int_time = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        int_signal = np.array([1, 2, 3, 2, 1], dtype=np.int32)

        time_norm, signal_norm = normalize_input_arrays(int_time, int_signal)

        # Should be NumPy arrays but may have promoted types
        self.assertIsInstance(time_norm, np.ndarray)
        self.assertIsInstance(signal_norm, np.ndarray)


class TestDataFrameExtraction(unittest.TestCase):
    """Test cases for DataFrame column extraction."""

    def setUp(self) -> None:
        """Set up DataFrame test fixtures."""
        time_data = np.linspace(0, 5, 51)
        acceleration_data = np.sin(2 * np.pi * time_data)
        velocity_data = -np.cos(2 * np.pi * time_data) / (2 * np.pi)

        self.df_standard = pd.DataFrame(
            {
                "time": time_data,
                "acceleration": acceleration_data,
                "velocity": velocity_data,
            }
        )

        # DataFrame with different column names
        self.df_custom = pd.DataFrame(
            {
                "t": time_data,
                "acc": acceleration_data,
                "vel": velocity_data,
                "pos": np.cumsum(velocity_data) * 0.1,
            }
        )

        # DataFrame with index as time
        self.df_index_time = pd.DataFrame(
            {"acceleration": acceleration_data, "velocity": velocity_data},
            index=time_data,
        )

    def test_extract_dataframe_columns_standard_names(self) -> None:
        """Test extraction with standard column names."""
        time_col, signal_col = extract_dataframe_columns(
            self.df_standard, independent_column="time", dependent_column="acceleration"
        )

        self.assertIsInstance(time_col, np.ndarray)
        self.assertIsInstance(signal_col, np.ndarray)
        self.assertEqual(len(time_col), len(signal_col))

    def test_extract_dataframe_columns_custom_names(self) -> None:
        """Test extraction with custom column names."""
        time_col, signal_col = extract_dataframe_columns(
            self.df_custom, independent_column="t", dependent_column="acc"
        )

        self.assertIsInstance(time_col, np.ndarray)
        self.assertIsInstance(signal_col, np.ndarray)
        np.testing.assert_array_equal(time_col, self.df_custom["t"].values)
        np.testing.assert_array_equal(signal_col, self.df_custom["acc"].values)

    def test_extract_dataframe_columns_index_as_time(self) -> None:
        """Test extraction using DataFrame index as independent variable."""
        time_col, signal_col = extract_dataframe_columns(
            self.df_index_time,
            independent_column=None,  # Use index
            dependent_column="acceleration",
        )

        self.assertIsInstance(time_col, np.ndarray)
        self.assertIsInstance(signal_col, np.ndarray)
        np.testing.assert_array_equal(time_col, self.df_index_time.index.values)
        np.testing.assert_array_equal(
            signal_col, self.df_index_time["acceleration"].values
        )

    def test_extract_dataframe_columns_missing_column(self) -> None:
        """Test error handling for missing columns."""
        with self.assertRaises(KeyError, msg="Should raise error for missing column"):
            extract_dataframe_columns(
                self.df_standard, independent_column="time", dependent_column="nonexistent"
            )

    def test_extract_dataframe_columns_auto_detection(self) -> None:
        """Test automatic detection of independent and dependent variable columns."""
        # This should try to automatically identify appropriate columns
        time_col, signal_col = extract_dataframe_columns(
            self.df_standard, independent_column="auto", dependent_column="auto"
        )

        self.assertIsInstance(time_col, np.ndarray)
        self.assertIsInstance(signal_col, np.ndarray)
        self.assertEqual(len(time_col), len(signal_col))

    def test_extract_dataframe_columns_scientific_naming_detection(self) -> None:
        """Test automatic detection of scientific column names 'independent' and 'dependent'."""
        # Create DataFrame with scientific naming convention and distractors
        df_scientific = pd.DataFrame({
            'independent': np.linspace(0, 10, 100),
            'dependent': np.sin(np.linspace(0, 10, 100)),
            'time': np.linspace(5, 15, 100),  # Distractor that might be chosen instead
            'data': np.cos(np.linspace(0, 10, 100)),  # Another distractor
            'noise': np.random.randn(100)
        })
        
        # Auto-detection should prioritize 'independent' over 'time' and 'dependent' over 'data'
        time_col, signal_col = extract_dataframe_columns(
            df_scientific, independent_column="auto", dependent_column="auto"
        )

        self.assertIsInstance(time_col, np.ndarray)
        self.assertIsInstance(signal_col, np.ndarray)
        self.assertEqual(len(time_col), 100)
        self.assertEqual(len(signal_col), 100)
        
        # Verify the correct columns were selected (should prefer scientific names)
        np.testing.assert_array_equal(time_col, df_scientific['independent'].values)
        np.testing.assert_array_equal(signal_col, df_scientific['dependent'].values)


class TestFlexibleInputIntegration(unittest.TestCase):
    """Integration tests for flexible input support across all functions."""

    def setUp(self) -> None:
        """Set up test data in multiple formats."""
        # Base data - need enough points for trigger detection (>40 for baseline + signal)
        np.random.seed(42)  # For reproducible tests
        self.time_data = np.linspace(0, 4, 81)  # Increased length

        # Create signal with clear baseline and event
        baseline_length = 30
        baseline = np.random.normal(0, 0.05, baseline_length)  # Small noise
        event = 2.0 * np.exp(
            -np.linspace(0, 3, len(self.time_data) - baseline_length)
        )  # Clear event
        self.signal_data = np.concatenate([baseline, event])

        # Multiple format representations
        self.formats = {
            "numpy_arrays": (self.time_data, self.signal_data),
            "pandas_series": (pd.Series(self.time_data), pd.Series(self.signal_data)),
            "python_lists": (self.time_data.tolist(), self.signal_data.tolist()),
            "dataframe": pd.DataFrame(
                {"time": self.time_data, "signal": self.signal_data}
            ),
        }

    def test_trigger_detection_all_input_formats(self) -> None:
        """Test trigger detection works with all input formats."""
        from src.series_utilities.core.trigger_detection import detect_event_trigger

        results = {}

        # Test array/series formats
        for name, (time_data, signal_data) in self.formats.items():
            if name != "dataframe":  # Skip DataFrame for this function
                with self.subTest(format=name):
                    trigger_idx = detect_event_trigger(time_data, signal_data)
                    results[name] = trigger_idx

        # All results should be similar (within tolerance)
        base_result = results["numpy_arrays"]
        for name, result in results.items():
            if (
                name != "numpy_arrays"
                and result is not None
                and base_result is not None
            ):
                self.assertAlmostEqual(
                    result,
                    base_result,
                    delta=2,
                    msg=f"Trigger detection should be consistent across formats: {name}",
                )

    def test_dc_removal_all_input_formats(self) -> None:
        """Test DC removal works with all input formats."""
        from src.series_utilities.core.dc_removal import remove_dc_component

        results = {}

        # Test array/series formats
        for name, (time_data, signal_data) in self.formats.items():
            if name != "dataframe":  # Skip DataFrame for this function
                with self.subTest(format=name):
                    _, corrected = remove_dc_component(time_data, signal_data, method='integral_mean')
                    # Check that integral is approximately zero
                    integral = np.trapezoid(corrected, self.time_data)
                    results[name] = integral

        # All integrals should be approximately zero
        for name, integral in results.items():
            self.assertAlmostEqual(
                integral,
                0.0,
                places=10,
                msg=f"DC removal should work consistently across formats: {name}",
            )

    def test_input_validation_comprehensive(self) -> None:
        """Test comprehensive input validation across formats."""
        for name, (time_data, signal_data) in self.formats.items():
            if name != "dataframe":  # Skip DataFrame for basic validation
                with self.subTest(format=name):
                    result = validate_input_data(time_data, signal_data)
                    self.assertTrue(result, f"Should validate format: {name}")


if __name__ == "__main__":
    unittest.main()
