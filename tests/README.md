# ModernAraBERT Test Suite

Comprehensive test suite for validating the ModernAraBERT repository components.

## Overview

This test suite provides unit and integration tests for:

- **Data Preprocessing** (`test_data_preprocessing.py`)
  - Arabic text normalization (diacritics, tatweel removal)
  - Text filtering (English detection, word counting)
  - File processing utilities
  - Edge cases and error handling

- **Tokenizer Extension** (`test_tokenizer.py`)
  - Vocabulary analysis and frequency counting
  - Token selection and filtering
  - Segmentation marker handling
  - Memory-efficient processing
  - Vocabulary extension logic

- **Benchmarking** (`test_benchmarking.py`)
  - Sentiment Analysis dataset loading
  - NER dataset structure validation
  - Label mapping and alignment
  - Evaluation metrics (Macro-F1, Micro-F1)
  - Memory tracking
  - Results export

## Running Tests

### Run All Tests

```bash
# From repository root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report (requires pytest-cov)
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Files

```bash
# Data preprocessing tests
pytest tests/test_data_preprocessing.py

# Tokenizer tests
pytest tests/test_tokenizer.py

# Benchmarking tests
pytest tests/test_benchmarking.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_data_preprocessing.py::TestArabicNormalization

# Run specific test function
pytest tests/test_data_preprocessing.py::TestArabicNormalization::test_remove_diacritics
```

### Run Tests by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Structure

### Fixtures (`conftest.py`)

Common test fixtures available to all test modules:

- `temp_dir`: Temporary directory for test files
- `sample_arabic_text`: Sample Arabic text
- `sample_arabic_text_with_diacritics`: Text with diacritics
- `sample_arabic_text_with_tatweel`: Text with elongation
- `sample_mixed_text`: Arabic-English mixed text
- `sample_text_file`: Sample text file
- `sample_sentiment_data`: SA test data
- `sample_ner_data`: NER test data

### Test Organization

Each test file is organized into test classes:

```python
class TestFeatureName:
    """Test suite for specific feature."""
    
    def test_specific_behavior(self):
        """Test description."""
        # Arrange
        # Act
        # Assert
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<behavior>`

### Example Test

```python
def test_normalize_arabic_text(sample_arabic_text_with_diacritics):
    """Test Arabic text normalization."""
    from src.pretraining.data_preprocessing import normalize_arabic_text
    
    # Act
    result = normalize_arabic_text(sample_arabic_text_with_diacritics)
    
    # Assert
    assert "َ" not in result  # No diacritics
    assert len(result) < len(sample_arabic_text_with_diacritics)
```

## Test Coverage

Current test coverage:

| Module | Coverage | Tests |
|--------|----------|-------|
| Data Preprocessing | ~80% | 25+ tests |
| Tokenizer Extension | ~75% | 20+ tests |
| Benchmarking | ~70% | 30+ tests |

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=src --cov-report=xml
```

## Dependencies

Required for running tests:

```bash
pip install pytest pytest-cov
```

Optional dependencies:

```bash
pip install pytest-xdist  # Parallel test execution
pip install pytest-timeout  # Test timeouts
```

## Known Limitations

- Some tests require specific environment setup (e.g., Farasa segmenter)
- Integration tests may be slow without mocking
- Full benchmarking tests require actual datasets

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass before committing
3. Aim for >80% code coverage
4. Add appropriate markers to tests
5. Update this README if adding new test categories

## Troubleshooting

### Import Errors

If you get import errors, ensure the repository root is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Fixture Not Found

Ensure `conftest.py` is in the `tests/` directory and contains the required fixtures.

### Tests Fail with Path Issues

Run tests from the repository root:

```bash
cd /path/to/modernbert-refactored
pytest tests/
```

---

**For detailed documentation on each test module, see the docstrings in the respective test files.**

