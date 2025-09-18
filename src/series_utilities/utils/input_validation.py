"""
Input validation utilities for series data.

Provides flexible input support for NumPy arrays, Pandas Series,
Pandas DataFrames with dependent/independent column pairs, and Python lists.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def validate_input_data(
    independent_data: Union[np.ndarray, pd.Series, List],
    dependent_data: Union[np.ndarray, pd.Series, List],
) -> bool:
    """
    Validate input data arrays for consistent length and type.

    Args:
        independent_data: Independent variable data (array-like)
        dependent_data: Dependent variable data (array-like)

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If validation fails
    """
    if len(independent_data) != len(dependent_data):
        raise ValueError("independent_data and dependent_data must have the same length")

    if len(independent_data) == 0:
        raise ValueError("Input data cannot be empty")

    return True


def normalize_input_arrays(
    independent_data: Union[np.ndarray, pd.Series, List],
    dependent_data: Union[np.ndarray, pd.Series, List],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize input data to NumPy arrays.

    Converts various input types (NumPy arrays, Pandas Series, Python lists)
    to consistent NumPy array format for processing.

    Args:
        independent_data: Independent variable data (array-like)
        dependent_data: Dependent variable data (array-like)

    Returns:
        Tuple of (independent_array, dependent_array) as NumPy arrays

    Raises:
        ValueError: If inputs cannot be converted or have mismatched lengths
    """
    # Validate first
    validate_input_data(independent_data, dependent_data)

    # Convert to NumPy arrays
    if isinstance(independent_data, pd.Series):
        independent_array = np.array(independent_data.values)
    elif isinstance(independent_data, (list, tuple)):
        independent_array = np.array(independent_data)
    else:
        independent_array = np.array(independent_data)

    if isinstance(dependent_data, pd.Series):
        dependent_array = np.array(dependent_data.values)
    elif isinstance(dependent_data, (list, tuple)):
        dependent_array = np.array(dependent_data)
    else:
        dependent_array = np.array(dependent_data)

    return independent_array, dependent_array


def extract_dataframe_columns(
    df: pd.DataFrame,
    independent_column: Optional[str] = "auto",
    dependent_column: Optional[str] = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract independent and dependent variable columns from a pandas DataFrame.

    Args:
        df: pandas DataFrame containing series data
        independent_column: Name of independent variable column, or None to use 
            index, or 'auto' for auto-detection
        dependent_column: Name of dependent variable column, or 'auto' for 
            auto-detection

    Returns:
        Tuple of (independent_array, dependent_array) as NumPy arrays

    Raises:
        KeyError: If specified columns don't exist
        ValueError: If auto-detection fails
    """
    # Handle independent variable column
    if independent_column is None:
        # Use DataFrame index as independent variable
        independent_array = np.array(df.index.values)
    elif independent_column == "auto":
        # Auto-detect independent variable column (prioritize scientific naming)
        independent_candidates = ["independent", "time", "t", "Time", "timestamp", "x"]
        independent_column_found = None
        for candidate in independent_candidates:
            if candidate in df.columns:
                independent_column_found = candidate
                break

        if independent_column_found is None:
            # Use index if no independent variable column found
            independent_array = np.array(df.index.values)
        else:
            independent_array = np.array(df[independent_column_found].values)
    else:
        if independent_column not in df.columns:
            raise KeyError(f"Independent variable column '{independent_column}' not found in DataFrame")
        independent_array = np.array(df[independent_column].values)

    # Handle dependent variable column
    if dependent_column == "auto":
        # Auto-detect dependent variable column (exclude independent column and use first remaining)
        excluded_columns = set()
        if independent_column and independent_column != "auto" and independent_column in df.columns:
            excluded_columns.add(independent_column)

        # Prefer common dependent variable names (prioritize scientific naming)
        dependent_candidates = [
            "dependent",
            "acceleration",
            "velocity",
            "position",
            "signal",
            "data",
            "value",
            "y",
        ]
        dependent_column_found = None

        for candidate in dependent_candidates:
            if candidate in df.columns and candidate not in excluded_columns:
                dependent_column_found = candidate
                break

        if dependent_column_found is None:
            # Use first column that's not the independent variable column
            for col in df.columns:
                if col not in excluded_columns:
                    dependent_column_found = col
                    break

        if dependent_column_found is None:
            raise ValueError("Could not auto-detect dependent variable column in DataFrame")

        dependent_array = np.array(df[dependent_column_found].values)
    else:
        if dependent_column not in df.columns:
            raise KeyError(f"Dependent variable column '{dependent_column}' not found in DataFrame")
        dependent_array = np.array(df[dependent_column].values)

    return independent_array, dependent_array


def get_input_info(data: Union[np.ndarray, pd.Series, pd.DataFrame, List]) -> Dict[str, Any]:
    """
    Get information about input data type and properties.

    Useful for debugging and understanding data characteristics.

    Args:
        data: Input data of any supported type

    Returns:
        Dictionary with data type information
    """
    info = {
        "type": type(data).__name__,
        "length": len(data) if hasattr(data, "__len__") else None,
        "shape": getattr(data, "shape", None),
        "dtype": getattr(data, "dtype", None),
    }

    if isinstance(data, pd.DataFrame):
        info["columns"] = list(data.columns)
        info["index_name"] = data.index.name
    elif isinstance(data, pd.Series):
        info["name"] = data.name
        info["index_name"] = data.index.name
    elif isinstance(data, np.ndarray):
        info["ndim"] = data.ndim

    return info
