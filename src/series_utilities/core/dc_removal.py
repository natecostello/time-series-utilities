"""
DC component removal using multiple methods.

Provides DC offset removal with three methods:
- 'integral_mean': Time-weighted integration (legacy, NO practical advantage)
- 'constant': Mean removal via scipy.signal.detrend (RECOMMENDED for all cases)
- 'linear': Linear detrending via scipy.signal.detrend (best when drift present)

**CRITICAL FINDING**: Analysis shows 'integral_mean' is inferior for ΔV=0 constraint
in ALL cases (uniform, non-uniform, with/without drift). It achieves ΔV ~1e-3 to 1e-5
while scipy methods achieve ΔV ~1e-16 (100,000× better).

**RECOMMENDATION**: Use scipy.signal.detrend() directly. This module adds no value for
DC removal. The 'integral_mean' method is maintained only for backward compatibility.
"""

from typing import Tuple, Union, Literal

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def remove_dc_component(
    independent_data: Union[np.ndarray, pd.Series],
    dependent_data: Union[np.ndarray, pd.Series],
    method: Literal['integral_mean', 'constant', 'linear'] = 'linear',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove DC component using specified method.

    Methods:
        'integral_mean': corrected_signal(t) = original_signal(t) - (1/T) ∫ original_signal(t) dt
            - Time-weighted integration via trapezoidal rule
            - Enforces ∫signal dt = 0 (NOT the same as ΔV=0!)
            - ΔV performance: POOR in all cases (~1e-3 to 1e-5)
            - Maintained only for backward compatibility
            - NO ADVANTAGE over scipy methods, even for non-uniform sampling

        'constant': corrected_signal = original_signal - mean(original_signal)
            - Mean removal via scipy.signal.detrend(type='constant')
            - ΔV performance: EXCELLENT in all cases (~1e-16)
            - Works for uniform AND non-uniform sampling
            - **RECOMMENDED for all use cases**

        'linear': corrected_signal = original_signal - (a*t + b)
            - Linear trend removal via scipy.signal.detrend(type='linear')
            - ΔV performance: EXCELLENT (~1e-15)
            - Best when linear drift/sensor bias present
            - **RECOMMENDED when drift detected**

    Performance Comparison (from analysis):
        - Uniform sampling, no drift: scipy 100× better ΔV (1e-16 vs 5e-5)
        - Uniform sampling, with drift: scipy 100,000× better ΔV (2e-16 vs 3e-5)
        - Non-uniform sampling: scipy STILL 100,000× better ΔV (6e-17 vs 6e-3)
        - CONCLUSION: integral_mean has NO advantage for ΔV=0 constraint

    Args:
        independent_data: Independent variable (e.g., time) array or Series
        dependent_data: Dependent variable (signal) array or Series
        method: DC removal method - 'linear' (default), 'constant', or 'integral_mean'

    Returns:
        Tuple of (independent_array, corrected_dependent_array)

    Raises:
        ValueError: If input arrays have different lengths, are empty, or method invalid

    Notes:
        - Default changed to 'linear' (most robust for typical signals)
        - 'integral_mean' available ONLY for backward compatibility
        - **Use scipy.signal.detrend() directly** - this function adds no value
        - See notebooks/dc_removal_methods_comparison.ipynb for detailed analysis
        - Consider this function deprecated; use scipy.signal.detrend instead
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

    # Validate method parameter
    valid_methods = ['integral_mean', 'constant', 'linear']
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {valid_methods}"
        )

    # Handle single point case
    if len(dependent_array) == 1:
        corrected_dependent = np.array([0.0])
        return independent_array, corrected_dependent

    # Apply selected DC removal method
    if method == 'integral_mean':
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

    elif method == 'constant':
        # Use scipy's constant detrending (mean removal)
        corrected_dependent = scipy_signal.detrend(dependent_array, type='constant')

    elif method == 'linear':
        # Use scipy's linear detrending (removes DC + linear drift)
        corrected_dependent = scipy_signal.detrend(dependent_array, type='linear')

    return independent_array, corrected_dependent
