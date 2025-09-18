# Python Package Development Guidelines for AI Agents

This document provides comprehensive guidelines for AI agents developing Python packages. Follow these standards to ensure high-quality, maintainable, and professional Python code.

## Package Description Template

**Package Name**: `series_utilities`
**Version**: `0.1.0`
**Description**: `A Python package for preprocessing and manipulating series data with dependent variables (e.g., values) and independent variables (e.g, time).`
**Main Purpose**: `Preprocessing and cleaning of series data, including transient removal and baseline correction for signal analysis applications.`
**Target Users**: `Data scientists, engineers, and researchers working with series data, particularly those analyzing signal events, vibration data, or other dependent/independent variable relationships.`
**Key Features**: 
- `Automatic event trigger detection and pre-event removal - detects signal onset using configurable statistical methods and removes preceding baseline data`
  - `Conservative approach: Use higher threshold multipliers (4-6σ instead of 2-3σ) for event detection`
  - `Bidirectional detection: Monitor absolute value or both positive/negative thresholds for signal events`
  - `Flat baseline assumption: Can use simple statistical methods since no detrending needed`
  - `Parameter naming: Use descriptive parameters like trigger_method='auto', sensitivity='conservative'`
- `DC component removal through integral mean subtraction - eliminates drift and enforces zero net change constraint by removing the time-averaged integral of the signal: corrected_signal(t) = original_signal(t) - (1/T) ∫ original_signal(t) dt`
- `Flexible input support for both NumPy arrays and Pandas DataFrames with dependent/independent column pairs`

## Required Standards and Practices

### 1. Python Coding Standards Compliance

All code must comply with the following standards:

- **PEP 8** - Style Guide for Python Code
- **PEP 257** - Docstring Conventions
- **PEP 484** - Type Hints (Python 3.5+)
- **PEP 20** - The Zen of Python principles

**Tools for Compliance**:
- Use `black` for code formatting
- Use `flake8` for linting
- Use `mypy` for static type checking
- Use `isort` for import sorting

### 2. Test-Driven Development (TDD)

Follow TDD methodology:

1. **Red** - Write a failing test first
2. **Green** - Write minimal code to pass the test
3. **Refactor** - Improve code while keeping tests passing

**Testing Requirements**:
- Minimum 90% code coverage
- All public methods must have tests
- Edge cases and error conditions must be tested
- Tests must be deterministic and independent

### 3. Package Structure

Use the following standard package structure:

```
[package_name]/
├── README.md
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── src/
│   └── [package_name]/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── [module].py
│       └── utils/
│           ├── __init__.py
│           └── [utility_module].py
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── test_[module].py
│   └── utils/
│       ├── __init__.py
│       └── test_[utility_module].py
├── docs/
│   ├── README.md
│   └── [documentation_files].md
├── examples/
│   ├── demo.ipynb
│   └── [example_scripts].py
└── notebooks/
    └── package_demonstration.ipynb
```

### 4. Test Organization

Each Python module must have a corresponding test file:

- For `src/package_name/module.py` → `tests/test_module.py`
- For `src/package_name/core/feature.py` → `tests/core/test_feature.py`
- Mirror the source directory structure in the tests directory
- One test file per module - consolidate all tests for a module in its single test file

### 5. Unit Testing with unittest

Use Python's built-in `unittest` framework:

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

class TestModuleName(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_function_name_expected_behavior(self):
        """Test description following naming convention."""
        # Arrange
        # Act
        # Assert
        pass
    
    @patch('module.dependency')
    def test_function_with_mock(self, mock_dependency):
        """Test with mocked dependencies."""
        pass

if __name__ == '__main__':
    unittest.main()
```

**Testing Commands**:
```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_module

# Run with coverage
python -m coverage run -m unittest discover tests/
python -m coverage report
python -m coverage html
```

### 6. Virtual Environment Setup

Always use virtual environments:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Deactivate when done
deactivate
```

**Environment Files**:
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies (testing, linting, etc.)
- `.env.example` - Template for environment variables

### 7. Git Ignore Configuration

Use this comprehensive `.gitignore`:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Custom
data/
logs/
temp/
```

### 8. Jupyter Notebook for Demonstration

Create `notebooks/package_demonstration.ipynb` that includes:

1. **Installation instructions**
2. **Import statements and setup**
3. **Basic usage examples**
4. **Advanced feature demonstrations**
5. **Real-world use cases**
6. **Performance benchmarks (if applicable)**
7. **Troubleshooting common issues**

**CRITICAL: Kernel Restart Requirements**:
- **Always restart the notebook kernel after modifying package source code**
- Use: `Kernel → Restart` or `Kernel → Restart & Clear Output` in Jupyter/VS Code
- Python imports are cached - kernel restart is required to reload changed modules
- Add this guidance prominently in notebook cells when demonstrating development workflow
- Test notebooks must be re-executed from top after any source code changes

**Notebook Structure Template**:
```markdown
# [Package Name] Demonstration

## 1. Installation
## 2. Basic Usage
## 3. Core Features
## 4. Advanced Examples
## 5. Integration Examples
## 6. Performance Analysis
## 7. Troubleshooting
```

### 9. Git Usage Requirements

Follow these Git practices:

- Initialize repository: `git init`
- Use meaningful commit messages following [Conventional Commits](https://www.conventionalcommits.org/)
- Create feature branches: `git checkout -b feature/feature-name`
- Use pull requests for code review
- Tag releases: `git tag -a v1.0.0 -m "Version 1.0.0"`

**Commit Message Format**:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Additional Recommended Practices

### Configuration Files

Include these configuration files:

**`pyproject.toml`**:
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "[package_name]"
version = "[version]"
description = "[description]"
authors = [{name = "[author]", email = "[email]"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

### Documentation Standards

Include comprehensive documentation:

- **README.md** with installation, usage, and examples
- **CHANGELOG.md** for version history
- **LICENSE** file


## Checklist for Agents

Before considering a Python package complete, ensure:

- [ ] All code follows PEP 8 standards
- [ ] TDD methodology was followed
- [ ] Test coverage is ≥90%
- [ ] All modules have corresponding test files
- [ ] Virtual environment is set up and documented
- [ ] .gitignore file is comprehensive
- [ ] Demonstration Jupyter notebook is complete and all cells run without errors
- [ ] Git repository is properly initialized
- [ ] Package structure follows standards
- [ ] Documentation is comprehensive
- [ ] Type hints are used throughout
- [ ] Error handling is robust
- [ ] Performance is acceptable
- [ ] Security considerations are addressed

## Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PEP 257 Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
- [Python Packaging Guide](https://packaging.python.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Test-Driven Development Guide](https://testdriven.io/test-driven-development/)