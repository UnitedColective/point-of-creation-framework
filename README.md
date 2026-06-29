# Point of Creation Framework (PCF)

**A Unified Framework for Generative Self-Governance**

By A.M. Sterling (Independent Researcher)  
Contact: ucroutreach@proton.me

## Overview

The Point of Creation Framework is a domain-agnostic metric for detecting generative self-governance in time series data. This preprint introduces the δ metric and applies it across four scientific domains: prime gaps, cardiac dynamics, neural activity, and circadian gene expression.

### Core Formula

```
δ = ρ₁(Δ²X) + 2/3
```

Where:
- **Δ²X** = second difference of any observable scalar sequence
- **ρ₁** = lag-1 autocorrelation
- **Expected value under white noise** = -2/3
- **δ > 0** = structure above white-noise floor (evidence of generative dynamics)
- **δ ≈ 0** = collapse to white-noise floor

## Key Findings

### 1. Prime Gaps (3×10⁸ primes)
- **δ = +0.154** [95% CI: +0.152, +0.156]
- **Status**: Confirmatory (most robust result)
- Stable across sample sizes (N = 10⁵ to 3×10⁸)
- Clockwise rotation bias in mod-24 residue transitions (22.1% vs 12.5%, p < 0.01)

### 2. Cardiac Dynamics (PhysioNet HRV)
- **Healthy subjects (n=54)**: δ̄ = +0.029 ± 0.065 (p = 0.002) ✓ above floor
- **CHF patients (n=29)**: δ̄ = +0.016 ± 0.086 (p = 0.33) ~ not significant
- **Status**: Directional trend (requires larger cohorts for confirmation)
- Between-group: p ≈ 0.47 (not significant, high variance)

### 3. Neural Metastability (iEEG)
- **Hybrid attractor state**: S ~ f^1.746 [95% CI: 1.52, 1.97]
- **Status**: Preliminary (3 subjects, 51 trial-subject pairs)
- R² = 0.858; superlinear scaling confirmed

### 4. Circadian Gene Expression (Supporting Evidence)
- **Cancer vs. normal cells**: MCF7 (0.7511) vs. MCF10A (0.7438)
- **Status**: Exploratory (counterintuitive: cancer shows rigidification)
- p = 4.24×10⁻⁴⁴

## Quick Start

### Installation

```bash
git clone https://github.com/UnitedColective/potential-adventure.git
cd potential-adventure
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from code.pcf_core import compute_pcf

# Your time series data
X = np.array([1.0, 1.2, 1.5, 1.3, 1.4, 1.6, ...])  # Any time series

# Compute PCF metric
result = compute_pcf(X, ci=0.95, n_bootstrap=1000)

print(f"δ = {result['delta']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"Above white-noise floor: {result['above_floor']}")
```

### Run Tests

```bash
python -m pytest tests/test_pcf_core.py -v
```

## Repository Structure

```
potential-adventure/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── PAPER.md                     # Full research paper
├── CONTRIBUTING.md              # Contribution guidelines
├── CITATION.md                  # How to cite this work
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
├── code/
│   └── pcf_core.py             # Core PCF implementation
├── tests/
│   └── test_pcf_core.py        # Unit tests
└── docs/
    └── METHODOLOGY.md           # Technical methodology
```

## Documentation

- **PAPER.md** - Complete research paper with all analyses, methods, and results
- **docs/METHODOLOGY.md** - Technical details on mathematical foundations, statistical testing, and domain-specific notes
- **CONTRIBUTING.md** - Guidelines for code standards, testing, and contributions
- **CITATION.md** - APA, BibTeX, and MLA citation formats

## Methodology Summary

### Statistical Testing
- **Surrogate null models**: Shuffle and phase-randomized (current: n=100, target: n≥1000)
- **Confidence intervals**: Block bootstrap (blocksize = √N)
- **Robustness checks**: Detrending, winsorizing, windowed analysis

### Data Sources
- **Prime gaps**: Open mathematical dataset (3×10⁸ primes)
- **Cardiac RR intervals**: PhysioNet (Congestive Heart Failure & Normal Sinus Rhythm databases)
- **Neural iEEG**: Human Amygdala MUA sEEG FearVideo dataset (Fedele et al., 2020)
- **Gene expression**: GEO (GSE76368, GSE76369; MCF7 vs MCF10A)

## Current Status & Limitations

| Domain | Result | Evidence | Status | Notes |
|--------|--------|----------|--------|-------|
| Prime gaps | δ = +0.154 | Strong | Confirmatory | 3×10⁸ data points; stable |
| Cardiac HRV | NSR > CHF trend | Moderate | Directional | n=29-54; p=0.47 (not sig) |
| Neural iEEG | α = 1.746 | Exploratory | Preliminary | n=3; conceptual connection |
| Circadian | Rigidification | Supporting | Exploratory | 8 timepoints; not δ direct |

### Known Limitations
1. Small sample sizes (cardiac n=29-54; neural n=3)
2. Surrogate testing at n=100 (below 1,000 minimum)
3. Heterogeneous methods across domains
4. Limited clinical validation (not diagnostic yet)
5. No medication data for cardiac cohort

## Future Work

- [ ] Expand cardiac cohort (target: n > 500)
- [ ] Increase surrogate count to 1,000+
- [ ] Test additional domains (climate, finance, protein folding, language)
- [ ] Develop clinical decision thresholds
- [ ] Theoretical connection to information theory and criticality
- [ ] Multivariate extensions

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for:
- Code standards and style guides
- Testing requirements
- How to add new analyses
- Data contribution process

## Citation

**APA:**
```
Sterling, A.M. (2024). The Point of Creation: A Unified Framework for Generative 
Self-Governance. Independent Research. https://github.com/UnitedColective/potential-adventure
```

**BibTeX:**
```bibtex
@software{balderas2024pcf,
  title={The Point of Creation: A Unified Framework for Generative Self-Governance},
  author={Sterling, A.M.},
  year={2026},
  url={https://github.com/UnitedColective/potential-adventure}
}
```

## Zenodo Archive

**Concept DOI (all versions)**: [10.5281/zenodo.20954700](https://doi.org/10.5281/zenodo.20954701)  
**GitHub Repository**: https://github.com/UnitedColective/potential-adventure  
**Status**: v2 Preprint (submitted for academic review)

## License

MIT License - See LICENSE file for details

## Contact

**A.M. Sterling 
Independent Researcher  
Email: ucroutreach@proton.me

---

**Note**: This framework integrates heterogeneous analyses across four domains. Independent replication and larger sample sizes are needed before clinical or theoretical conclusions can be drawn. This is an exploratory preprint inviting academic discourse and replication.
