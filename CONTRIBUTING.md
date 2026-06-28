# Contributing to the Point of Creation Framework

Thank you for your interest in contributing! This document outlines how to contribute code, analysis, or ideas.

## Code of Conduct

- Be respectful and professional
- Acknowledge limitations and uncertainties
- Prioritize reproducibility and transparency
- Welcome diverse perspectives and methods

## How to Contribute

### 1. Report Issues or Suggest Improvements

- Open an issue on GitHub
- Describe the problem clearly with minimal reproducible example
- Link to relevant sections of the paper or code

### 2. Contribute Code

**Prerequisites:**
- Python 3.8+
- NumPy, SciPy, Pandas
- Git

**Process:**

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes:
   - Add docstrings to all functions
   - Follow PEP 8 style guide
   - Include unit tests for new functionality

4. Test your code:
   ```bash
   python code/pcf_core.py  # Run built-in tests
   pytest tests/            # Run full test suite
   ```

5. Commit with clear messages:
   ```bash
   git commit -m "Add [feature]: brief description of changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request (PR) with:
   - Clear description of changes
   - Link to any related issues
   - Explanation of methodology if applicable

### 3. Contribute Analyses

If you want to apply PCF to a new domain:

1. Create a new subdirectory under `analysis/`:
   ```
   analysis/your_domain/
   ├── README.md          # Domain-specific notes
   ├── preprocessing.py   # Data loading & cleaning
   ├── analysis.py        # Your analysis code
   └── results.ipynb      # Jupyter notebook with results
   ```

2. Document:
   - Data source and access instructions
   - Preprocessing steps
   - Statistical tests performed
   - Limitations and caveats

3. Submit a PR with your analysis notebook and findings

### 4. Contribute Data

If you have domain-specific data you'd like to analyze:

1. Check for privacy/ethical constraints
2. Contact maintainers to discuss data sharing
3. Provide metadata: variable descriptions, time resolution, sample size
4. Submit as an issue with data attachment or link

## Code Standards

### Style
- Follow PEP 8
- Use type hints where possible
- Keep functions small and focused
- Maximum line length: 100 characters

### Documentation
```python
def your_function(param1: np.ndarray, param2: int = 100) -> Dict:
    """
    Brief one-line description.
    
    Longer explanation of what the function does and why.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 100)
    
    Returns:
        Dictionary with keys 'result', 'confidence_interval'
    
    Raises:
        ValueError: If param2 < 0
    
    Example:
        >>> result = your_function(data, param2=50)
        >>> print(result['result'])
    """
```

### Testing

Add unit tests in `tests/`:

```python
import pytest
from code.pcf_core import compute_pcf

def test_white_noise():
    """PCF metric should be ~0 for white noise."""
    wn = np.random.randn(1000)
    result = compute_pcf(wn)
    assert abs(result['delta']) < 0.1, "White noise should have δ ≈ 0"

def test_ar1_positive():
    """PCF metric should be > 0 for AR(1) with positive autocorrelation."""
    ar1 = np.zeros(1000)
    ar1[0] = np.random.randn()
    for t in range(1, len(ar1)):
        ar1[t] = 0.7 * ar1[t-1] + np.random.randn()
    result = compute_pcf(ar1)
    assert result['delta'] > 0, "AR(1) should have δ > 0"
```

## Manuscript/Paper Contributions

If you want to contribute to the research:

1. Discuss ideas in issues first
2. Propose methodology in a PR to `docs/`
3. Run analyses on a feature branch
4. Submit results with full documentation of methods
5. Expect peer review and revision

## Questions?

Open an issue or contact the maintainers at ucroutreach@proton.me

## Acknowledgments

All contributors will be acknowledged in the project README. For major contributions, consider authorship discussion.

---

*Last updated: 2024*
