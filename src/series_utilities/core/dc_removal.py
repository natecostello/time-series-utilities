"""
DC component removal through integral mean subtraction.

Eliminates drift and enforces zero net change constraint by removing
the time-averaged integral of the signal.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def remove_dc_component(
    independent_data: Union[np.ndarray, pd.Series],
    dependent_data: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove DC component through integral mean subtraction.

    Implements: corrected_signal(t) = original_signal(t) - (1/T) ∫ original_signal(t) dt

    This enforces zero net change constraint (e.g., ΔV = 0 for acceleration signals)
    by removing the time-averaged integral of the signal.

    Args:
        independent_data: Independent variable array or Series
        dependent_data: Dependent variable (signal) array or Series

    Returns:
        Tuple of (independent_array, corrected_dependent_array)

    Raises:
        ValueError: If input arrays have different lengths or are empty
    """
    # Input validation
    if len(independent_data) != len(dependent_data):
        raise ValueError("independent_data and dependent_data must have the same length")

    if len(independent_data) == 0:
        raise ValueError("Input data cannot be empty")

    # Convert to numpy arrays for processing
    if isinstance(independent_data, pd.Series):
        independent_array = np.array(independent_data.values)
    else:
        independent_array = np.array(independent_data)

    if isinstance(dependent_data, pd.Series):
        dependent_array = np.array(dependent_data.values)
    else:
        dependent_array = np.array(dependent_data)

    # Handle single point case
    if len(dependent_array) == 1:
        corrected_dependent = np.array([0.0])
        return independent_array, corrected_dependent

    # Calculate the integral of the signal using trapezoidal rule
    # Note: Using trapezoid (newer) instead of deprecated trapz
    signal_integral = np.trapezoid(dependent_array, independent_array)

    # Calculate total time duration T
    T = independent_array[-1] - independent_array[0]

    # Handle zero duration case (all time points identical)
    if abs(T) < 1e-15:
        # If no time duration, just remove mean value
        integral_mean = np.mean(dependent_array)
    else:
        # Calculate integral mean: (1/T) ∫ signal(t) dt
        integral_mean = signal_integral / T

    # Apply correction: corrected_signal(t) = original_signal(t) - integral_mean
    corrected_dependent = dependent_array - integral_mean

    return independent_array, corrected_dependent
