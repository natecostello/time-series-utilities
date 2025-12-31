# Series Utilities

A Python package for preprocessing and manipulating series data with dependent variables (values) and independent variables (e.g., time, frequency, distance).

## Overview

Series Utilities provides essential tools for preprocessing and cleaning series data, with particular focus on transient removal and baseline correction for signal analysis applications. The package is designed for data scientists, engineers, and researchers working with time-series data, vibration analysis, or other dependent/independent variable relationships.

## Key Features

### 🎯 **Automatic Event Trigger Detection and Pre-Event Removal**
- Detects signal onset using configurable statistical methods 
- Removes preceding baseline data with intelligent baseline backup
- Conservative approach: Use higher threshold multipliers (4-6σ instead of 2-3σ) for event detection
- Bidirectional detection: Monitor absolute value or both positive/negative thresholds for signal events
- Flat baseline assumption: Uses simple statistical methods since no detrending needed
- Parameter naming: Descriptive parameters like `threshold_sigma=4.0`, `backup_sigma=2.0`

### 📀 **DC Component Removal** (⚠️ Deprecated - Use scipy.signal.detrend)
- Three methods: `'integral_mean'` (legacy), `'constant'` (recommended), `'linear'` (for drift)
- **CRITICAL**: Analysis shows scipy methods outperform in ALL cases (100,000× better ΔV=0)
- **integral_mean**: NO advantage, even for non-uniform sampling (ΔV ~1e-3)
- **constant/linear**: Excellent ΔV≈0 enforcement (ΔV ~1e-16) in all cases
- **RECOMMENDATION**: Use `scipy.signal.detrend()` directly, this feature adds no value

### 🔧 **Flexible Input Support**
- NumPy arrays and Pandas DataFrames 
- Dependent/independent column pairs
- Automatic column detection or custom column specification
- Consistent API across all input formats

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/your-username/time-series-utilities.git
cd time-series-utilities

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Requirements
- Python 3.8+
- NumPy ≥ 2.0.0
- Pandas ≥ 2.0.0
- SciPy ≥ 1.10.0

## Quick Start

### Basic Event Detection and Pre-Event Removal

```python
import numpy as np
from series_utilities import detect_event_trigger, remove_pre_event_data

# Create sample data: baseline noise + shock event
time = np.arange(100) * 0.01  # 100 samples at 0.01s intervals
baseline = np.random.normal(0, 0.1, 50)  # Baseline noise
event = 5.0 * np.exp(-np.linspace(0, 3, 50))  # Exponential decay event
signal = np.concatenate([baseline, event])

# Detect event trigger with intelligent baseline backup
trigger_idx = detect_event_trigger(
    time, 
    signal, 
    threshold_sigma=4.0,     # Conservative 4σ threshold
    backup_sigma=2.0         # 2σ for backing up to true onset
)
print(f"Event detected at index {trigger_idx} (t={time[trigger_idx]:.3f}s)")

# Remove pre-event data
filtered_time, filtered_signal = remove_pre_event_data(
    time, 
    signal, 
    trigger_index=trigger_idx,
    padding_samples=5,       # Keep 5 samples before trigger
    normalize_independent=True  # Reset time to start at 0
)
```

### DC Component Removal

```python
from series_utilities import remove_dc_component

# Method 1: Integral mean (default, best for non-uniform sampling)
time_corr, signal_corr = remove_dc_component(time, signal, method='integral_mean')

# Method 2: Constant (mean removal, best ΔV=0 for uniform sampling)
time_corr, signal_corr = remove_dc_component(time, signal, method='constant')

# Method 3: Linear (removes DC + drift)
time_corr, signal_corr = remove_dc_component(time, signal, method='linear')

# Verify zero integral for integral_mean method
integral = np.trapezoid(signal_corr, time_corr)
print(f"Signal integral after DC removal: {integral:.2e}")

# Note: For uniform sampling, scipy.signal.detrend is also recommended:
from scipy import signal
corrected = signal.detrend(signal, type='constant')  # or type='linear'
```

### Working with DataFrames

```python
import pandas as pd

# Create DataFrame with time and acceleration columns
df = pd.DataFrame({
    'time': time,
    'acceleration': signal
})

# Detect trigger and remove pre-event data in one step
filtered_df = remove_pre_event_data(
    df,
    trigger_index=trigger_idx,
    normalize_independent=True
)
```

### Parameter Sensitivity Analysis

```python
# Test different sensitivity settings
configs = [
    {'name': 'Conservative', 'threshold_sigma': 6.0, 'backup_sigma': 3.0},
    {'name': 'Moderate', 'threshold_sigma': 4.0, 'backup_sigma': 2.0}, 
    {'name': 'Sensitive', 'threshold_sigma': 3.0, 'backup_sigma': 1.5}
]

for config in configs:
    trigger_idx = detect_event_trigger(
        time, signal,
        threshold_sigma=config['threshold_sigma'],
        backup_sigma=config['backup_sigma']
    )
    print(f"{config['name']:>12}: t = {time[trigger_idx]:.4f}s (idx={trigger_idx})")
```

## API Reference

### Core Functions

#### `detect_event_trigger(time_data, signal_data, threshold_sigma=4.0, baseline_samples=20, backup_sigma=2.0)`

Detect the true onset of a signal event using statistical change detection with intelligent baseline backup.

**Parameters:**
- `time_data`: Independent variable array or Series
- `signal_data`: Dependent variable (signal) array or Series  
- `threshold_sigma` (float): Threshold multiplier for initial trigger detection (conservative, default 4.0)
- `baseline_samples` (int): Number of samples to use for baseline statistics (default 20)
- `backup_sigma` (float): Threshold for backing up to true onset (more sensitive, default 2.0)

**Returns:**
- `int` or `None`: Index of true event onset, or None if no event detected

#### `remove_pre_event_data(independent_var, dependent_var=None, trigger_index=0, padding_samples=0, normalize_independent=True)`

Remove pre-event data from signal, keeping data from trigger point onwards.

**Parameters:**
- `independent_var`: Time array/Series or DataFrame containing both time and signal
- `dependent_var`: Signal array/Series (None if independent_var is DataFrame)
- `trigger_index` (int): Index where true event begins
- `padding_samples` (int): Number of samples to keep before trigger point
- `normalize_independent` (bool): If True, reset independent axis to start at zero

**Returns:**
- `tuple` or `DataFrame`: Filtered time and signal arrays, or filtered DataFrame

#### `remove_dc_component(independent_data, dependent_data, method='integral_mean')` ⚠️ DEPRECATED

**⚠️  WARNING: This function is effectively deprecated. Use scipy.signal.detrend() directly.**

Remove DC component using specified method.

**Parameters:**
- `independent_data`: Independent variable (time) array or Series
- `dependent_data`: Dependent variable (signal) array or Series
- `method` (str): DC removal method - `'integral_mean'` (default), `'constant'`, or `'linear'`
  - **'integral_mean'**: Time-weighted integration (legacy, NO advantage)
    - ΔV performance: POOR (~1e-3 to 1e-5) in ALL cases
    - Maintained only for backward compatibility
  - **'constant'**: Mean removal via scipy.signal.detrend(type='constant') **RECOMMENDED**
    - ΔV performance: EXCELLENT (~1e-16) in ALL cases
    - Works for uniform AND non-uniform sampling
  - **'linear'**: Linear detrending via scipy.signal.detrend(type='linear') **RECOMMENDED**
    - ΔV performance: EXCELLENT (~1e-15)
    - Best when drift present

**Returns:**
- `tuple`: (independent_array, corrected_dependent_array)

**Performance Reality Check:**
- Uniform, no drift: scipy 100× better ΔV (1e-16 vs 5e-5)
- Uniform, with drift: scipy 100,000× better ΔV (2e-16 vs 3e-5)  
- Non-uniform: scipy STILL 100,000× better ΔV (6e-17 vs 6e-3)
- **CONCLUSION**: integral_mean has NO advantage in ANY case

**Recommendation:**
```python
# Instead of this package function:
from series_utilities import remove_dc_component
t, signal_corr = remove_dc_component(t, signal, method='constant')

# Just use scipy directly:
from scipy import signal
signal_corr = signal.detrend(signal, type='constant')  # or type='linear'
```

See [comparison notebook](notebooks/dc_removal_methods_comparison.ipynb) for detailed analysis.

## Examples and Demonstrations

See the [package demonstration notebook](notebooks/package_demonstration.ipynb) for comprehensive examples including:

- Sensitivity analysis across 3σ-6σ parameter ranges
- Comparison of different detection approaches
- Integration with DC removal workflows  
- Visualization of results
- Edge case handling

## Development

### Running Tests

The package uses Python's built-in `unittest` framework:

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test module
python -m unittest tests.core.test_trigger_detection -v

# Run with coverage (if coverage is installed)
python -m coverage run -m unittest discover tests/
python -m coverage report
python -m coverage html
```

### Code Quality

The package follows strict coding standards:

- **PEP 8** compliance via `black` and `flake8`
- **PEP 257** docstring conventions
- **PEP 484** type hints
- Minimum 90% test coverage

```bash
# Format code
python -m black src/ tests/
python -m isort src/ tests/

# Check compliance  
python -m flake8 src/ tests/ --max-line-length=88
python -m mypy src/
```

## Project Structure

```
series_utilities/
├── README.md
├── pyproject.toml          # Package configuration
├── requirements.txt        # Production dependencies  
├── requirements-dev.txt    # Development dependencies
├── src/
│   └── series_utilities/
│       ├── __init__.py
│       ├── core/
│       │   ├── trigger_detection.py  # Event detection algorithms
│       │   └── dc_removal.py         # DC component removal
│       └── utils/
│           └── input_validation.py   # Input handling utilities
├── tests/
│   ├── core/
│   │   ├── test_trigger_detection.py
│   │   └── test_dc_removal.py
│   └── utils/
│       └── test_input_validation.py
├── notebooks/
│   └── package_demonstration.ipynb   # Comprehensive examples
└── docs/
    └── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow TDD methodology (Red-Green-Refactor)
- Maintain test coverage ≥ 90%
- Use descriptive commit messages
- Update documentation for new features
- Follow existing code style and patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in research, please cite:

```bibtex
@software{series_utilities,
  title = {Series Utilities: Python Package for Time-Series Data Preprocessing},
  author = {Your Name},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/your-username/time-series-utilities}
}
```

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-username/time-series-utilities/issues)  
- 💬 [Discussions](https://github.com/your-username/time-series-utilities/discussions)