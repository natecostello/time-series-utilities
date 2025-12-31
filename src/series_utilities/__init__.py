"""
Series Utilities Package

A Python package for preprocessing and manipulating series data with dependent
variables (values) and independent variables (e.g., time, frequency, distance).

This package provides tools for:
- Automatic event trigger detection and pre-event removal
- DC component removal through integral mean subtraction
- Flexible input support for NumPy arrays and Pandas DataFrames
"""

__version__ = "0.3.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.dc_removal import remove_dc_component

# Import main functionality
from .core.trigger_detection import detect_event_trigger, remove_pre_event_data
from .utils.input_validation import (
    extract_dataframe_columns,
    get_input_info,
    normalize_input_arrays,
    validate_input_data,
)

__all__ = [
    "detect_event_trigger",
    "remove_pre_event_data",
    "remove_dc_component",
    "validate_input_data",
    "normalize_input_arrays",
    "extract_dataframe_columns",
    "get_input_info",
]
