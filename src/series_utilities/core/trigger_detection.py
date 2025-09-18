"""
Trigger detection and pre-event removal functionality    if len(dependent_data) < baseline_samples * 2:
        raise ValueError("Signal too short for reliable baseline estimation")

    # Convert to NumPy arrays for processing
    if isinstance(dependent_data, pd.Series):
        signal_array = dependent_data.values
    else:
        signal_array = np.array(dependent_data)mentation follows conservative approach with bidirectional detection
and configurable statistical methods.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def detect_event_trigger(
    independent_data: Union[np.ndarray, pd.Series],
    dependent_data: Union[np.ndarray, pd.Series],
    threshold_sigma: float = 4.0,
    baseline_samples: int = 20,
    backup_sigma: float = 2.0,
) -> Optional[int]:
    """
    Detect the true onset of a signal event using statistical change detection
    with intelligent baseline backup.

    This function first detects the obvious trigger point, then intelligently
    backs up to find the actual event onset where the signal first departs
    from baseline noise levels.

    Conservative approach: Use higher threshold multipliers (4-6σ instead of
    2-3σ) for initial detection
    Bidirectional detection: Monitor absolute value or both positive/negative
    thresholds
    Intelligent baseline backup: From trigger point, backup until signal
    returns to baseline levels

    Args:
        independent_data: Independent variable array or Series
        dependent_data: Dependent variable (signal) array or Series
        threshold_sigma: Threshold multiplier for initial trigger detection
            (conservative)
        baseline_samples: Number of samples to use for baseline statistics
        backup_sigma: Threshold for backing up to true onset (more sensitive)

    Returns:
        Index of true event onset (backed up from initial trigger), or None
        if no event detected

    Raises:
        ValueError: If input arrays have different lengths or invalid parameters
    """
    # Input validation
    if len(independent_data) != len(dependent_data):
        raise ValueError("independent_data and dependent_data must have the same length")

    if threshold_sigma <= 0 or backup_sigma <= 0:
        raise ValueError("threshold_sigma and backup_sigma must be positive")

    if len(dependent_data) < baseline_samples * 2:
        raise ValueError("Signal too short for reliable baseline estimation")

    # Convert to numpy arrays for processing
    if isinstance(dependent_data, pd.Series):
        dependent_array = dependent_data.values
    else:
        dependent_array = np.array(dependent_data)

    # Calculate baseline statistics using initial samples
    baseline_data = dependent_array[:baseline_samples]
    baseline_mean = np.mean(baseline_data)
    baseline_std = np.std(baseline_data)

    # If baseline std is too small, no significant signal variation
    if baseline_std < 1e-10:
        return None

    # Calculate threshold for initial event detection (conservative)
    initial_threshold = threshold_sigma * baseline_std

    # Look for initial trigger point beyond baseline region
    search_start = baseline_samples
    initial_trigger_idx = None

    # Bidirectional detection - check for deviations in either direction
    for i in range(search_start, len(dependent_array)):
        signal_deviation = abs(dependent_array[i] - baseline_mean)

        if signal_deviation > initial_threshold:
            initial_trigger_idx = i
            break

    # No initial trigger found
    if initial_trigger_idx is None:
        return None

    # Simple incremental backup: Find FIRST point within baseline band
    # (closest to trigger)
    backup_threshold = backup_sigma * baseline_std

    # Start from initial trigger and back up incrementally
    true_onset_idx = initial_trigger_idx

    # Back up index by index to find the first point within baseline band
    for i in range(initial_trigger_idx - 1, search_start - 1, -1):
        signal_value = dependent_array[i]
        deviation_from_baseline = abs(signal_value - baseline_mean)

        # If this point is within our target band, this is our onset
        if deviation_from_baseline <= backup_threshold:
            true_onset_idx = i
            # Found the first point within baseline - stop here (don't keep going back)
            break

    # Ensure we don't back up into the baseline calibration region
    true_onset_idx = max(true_onset_idx, baseline_samples)

    return true_onset_idx


def remove_pre_event_data(
    independent_var: Union[np.ndarray, pd.Series, pd.DataFrame],
    dependent_var: Optional[Union[np.ndarray, pd.Series]] = None,
    trigger_index: int = 0,
    padding_samples: int = 0,
    normalize_independent: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
    """
    Remove pre-event data from signal, keeping data from trigger point onwards.

    The trigger_index should already represent the true event onset (as determined
    by detect_event_trigger with intelligent baseline backup).

    Args:
        independent_var: Time array/Series or DataFrame containing both time and signal
        dependent_var: Signal array/Series (None if independent_var is DataFrame)
        trigger_index: Index where true event begins (from detect_event_trigger)
        padding_samples: Number of samples to keep before trigger point
        normalize_independent: If True, reset independent axis to start at zero

    Returns:
        Filtered time and signal arrays, or filtered DataFrame

    Raises:
        ValueError: If trigger_index is invalid
    """
    # Handle DataFrame input
    if isinstance(independent_var, pd.DataFrame):
        if trigger_index is None:
            raise ValueError("trigger_index required for DataFrame processing")

        if trigger_index >= len(independent_var):
            raise ValueError("trigger_index exceeds data length")

        # Calculate start index with padding (no manual backup needed)
        start_index = max(0, trigger_index - padding_samples)

        # Filter DataFrame
        filtered_df = independent_var.iloc[start_index:].reset_index(drop=True)

        # Normalize independent axis if requested
        if normalize_independent and len(filtered_df) > 0:
            # Assume first column is the independent variable (time)
            if len(filtered_df.columns) > 0:
                first_column = filtered_df.columns[0]
                filtered_df = filtered_df.copy()  # Avoid SettingWithCopyWarning
                filtered_df[first_column] = (
                    filtered_df[first_column] - filtered_df[first_column].iloc[0]
                )

        return filtered_df

    # Handle array/Series input
    if dependent_var is None:
        raise ValueError("dependent_var required when independent_var is not DataFrame")

    if trigger_index is None:
        raise ValueError("trigger_index required")

    if trigger_index >= len(independent_var):
        raise ValueError("trigger_index exceeds data length")

    # Calculate start index with padding (no manual backup needed)
    start_index = max(0, trigger_index - padding_samples)

    # Convert to numpy arrays and filter
    if isinstance(independent_var, pd.Series):
        filtered_time = np.array(independent_var.values)[start_index:]
    else:
        filtered_time = np.array(independent_var)[start_index:]
    
    if isinstance(dependent_var, pd.Series):
        filtered_signal = np.array(dependent_var.values)[start_index:]
    else:
        filtered_signal = np.array(dependent_var)[start_index:]

    # Normalize independent axis to start at zero if requested
    if normalize_independent and len(filtered_time) > 0:
        filtered_time = filtered_time - filtered_time[0]

    return filtered_time, filtered_signal
