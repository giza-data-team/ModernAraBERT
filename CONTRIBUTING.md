# Contributing to ModernAraBERT

We welcome contributions to ModernAraBERT! This document provides guidelines for contributing to the project.

## Table of Contents

- [Contributing to ModernAraBERT](#contributing-to-modernarabert)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [How to Contribute](#how-to-contribute)
    - [Types of Contributions](#types-of-contributions)
  - [Development Setup](#development-setup)
  - [Code Style](#code-style)
    - [Formatting](#formatting)
    - [Type Hints](#type-hints)
    - [Docstrings](#docstrings)
  - [Testing](#testing)
    - [Running Tests](#running-tests)
    - [Writing Tests](#writing-tests)
  - [Pull Request Process](#pull-request-process)
    - [Pull Request Checklist](#pull-request-checklist)
  - [Reporting Bugs](#reporting-bugs)
    - [Bug Report Template](#bug-report-template)
  - [Suggesting Enhancements](#suggesting-enhancements)
    - [Enhancement Template](#enhancement-template)
  - [Repository Structure](#repository-structure)
  - [Areas for Contribution](#areas-for-contribution)
  - [Questions?](#questions)
  - [Recognition](#recognition)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR-USERNAME/ModernAraBERT.git
   cd ModernAraBERT
   ```

3. Add the upstream repository:

   ```bash
   git remote add upstream https://github.com/giza-data-team/ModernAraBERT.git
   ```

## How to Contribute

### Types of Contributions

- **Bug Reports**: Report bugs through GitHub issues
- **Feature Requests**: Suggest new features through GitHub issues
- **Code Contributions**: Submit bug fixes or new features via pull requests
- **Documentation**: Improve documentation, add examples, or fix typos
- **Testing**: Add or improve test coverage

## Development Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. Install pre-commit hooks (recommended):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We follow Python best practices and PEP 8 guidelines:

### Formatting

- **Black**: Use Black for code formatting (line length: 100)

  ```bash
  black src/ scripts/ tests/
  ```

- **isort**: Use isort for import sorting

  ```bash
  isort src/ scripts/ tests/
  ```

### Type Hints

- Add type hints to function signatures
- Use `typing` module for complex types

```python
from typing import List, Dict, Optional

def process_text(text: str, max_length: int = 512) -> Dict[str, List[int]]:
    """Process input text and return tokenized output.
    
    Args:
        text: Input text string
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing tokenized outputs
    """
    pass
```

### Docstrings

- Use Google-style docstrings
- Document all public functions, classes, and methods

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10
) -> Dict[str, float]:
    """Train the model on provided data.
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is negative
    """
    pass
```

## Testing

### Running Tests

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_preprocessing.py
```

Run with coverage:

```bash
pytest --cov=src tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<expected_behavior>`
- Test both success and failure cases
- Use fixtures for common test data

```python
import pytest
from src.pretraining.data_preprocessing import preprocess_text

def test_preprocess_text_removes_diacritics():
    """Test that diacritics are removed correctly."""
    input_text = "مَرْحَبًا"
    expected = "مرحبا"
    assert preprocess_text(input_text) == expected

def test_preprocess_text_empty_string():
    """Test handling of empty string input."""
    assert preprocess_text("") == ""
```

## Pull Request Process

1. **Create a branch** for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, concise code
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**:

   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

   Commit message format:
   - `feat: Add new feature`
   - `fix: Fix bug in X`
   - `docs: Update documentation`
   - `test: Add tests for Y`
   - `refactor: Refactor Z`
   - `style: Format code`

4. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear title and description
   - Link related issues using `#issue-number`

### Pull Request Checklist

- [ ] Code follows the style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated (if applicable)
- [ ] Tests added and passing
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear and descriptive

## Reporting Bugs

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Expected behavior** vs **actual behavior**
3. **Steps to reproduce**:

   ```
   1. Run command X
   2. With parameters Y
   3. See error Z
   ```

4. **Environment information**:
   - Python version
   - PyTorch version
   - transformers version
   - Operating system
5. **Error messages** or logs (use code blocks)
6. **Relevant code** or configuration

### Bug Report Template

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.x.x
- PyTorch version: x.x.x
- OS: Ubuntu 20.04

## Additional Context
Any other relevant information
```

## Suggesting Enhancements

When suggesting enhancements:

1. **Check existing issues** to avoid duplicates
2. **Provide clear use case** and motivation
3. **Describe the proposed solution**
4. **Consider alternatives** you've thought about
5. **Discuss potential impact** on existing functionality

### Enhancement Template

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
What other approaches were considered?

## Additional Context
Any other relevant information
```

## Repository Structure

Understanding the structure helps in making contributions:

```
ModernAraBERT/
├── src/                     # Source code
│   ├── pretraining/         # Pretraining modules
│   ├── benchmarking/        # Benchmarking modules
│   └── utils/               # Shared utilities
├── scripts/                 # Executable scripts
├── tests/                   # Test files
├── docs/                    # Documentation
└── configs/                 # Configuration files
```

## Areas for Contribution

We especially welcome contributions in:

- **Extending to new tasks**: Add support for more Arabic NLP tasks
- **Optimization**: Improve training/inference efficiency
- **Documentation**: Improve guides, add examples, translate docs
- **Benchmarking**: Add more benchmark datasets
- **Tests**: Increase test coverage
- **Multilingual**: Extend approach to other languages

## Questions?

If you have questions:

1. Check existing [documentation](./docs/)
2. Search [existing issues](https://github.com/giza-data-team/ModernAraBERT/issues)
3. Open a new issue with the `question` label
4. Contact maintainers via email (see README)

## Recognition

Contributors will be acknowledged in:

- README contributors section
- Release notes for significant contributions
- Paper acknowledgments (for substantial contributions)

Thank you for contributing to ModernAraBERT! 🎉
