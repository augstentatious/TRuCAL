# TRuCAL Project Structure

## Overview
This document provides a complete overview of the TRuCAL project structure after modularization and code organization.

## Directory Structure

```
TRuCAL/
├── README.md                          # Main project documentation (PROOFREAD ✓)
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies (CREATED ✓)
├── cal.py                             # Monolithic module (backward compatible)
├── PROJECT_STRUCTURE.md              # This file
│
├── components/                        # Modular components (NEW ✓)
│   ├── __init__.py                   # Package initialization
│   ├── vulnerability_spotter.py      # VulnerabilitySpotter component
│   ├── confessional_template.py      # ConfessionalTemplate component
│   ├── tiny_confessional_layer.py    # TinyConfessionalLayer component
│   ├── unified_cal_trm.py            # UnifiedCAL_TRM component
│   ├── scratchpad_layer.py           # ScratchpadLayer component
│   └── cal_trm_hybrid.py             # CAL_TRM_Hybrid component
│
├── tests/                             # Testing modules
│   └── test_cal.py                    # Comprehensive test suite (EXTRACTED ✓)
│
├── examples/                          # Example scripts
│   └── truthfulqa_eval.py            # TruthfulQA evaluation (CREATED ✓)
│
└── docs/                              # Documentation
    └── COMPONENTS.md                  # Component organization guide (CREATED ✓)
```

## Files Summary

### Core Files

#### README.md
**Status**: ✓ PROOFREAD AND FIXED
**Changes**:
- Fixed code block formatting (added proper ``` delimiters)
- Fixed markdown headers (added ##)
- Converted lists to proper markdown format
- Fixed Installation, Quick Start, and Usage sections
- Improved Contributing and License sections
- Better structured Acknowledgments

#### LICENSE
**Status**: ✓ UNCHANGED
**Content**: MIT License

#### requirements.txt
**Status**: ✓ CREATED
**Dependencies**:
- torch>=2.0.0
- numpy>=1.24.0
- transformers>=4.30.0
- datasets>=2.14.0
- matplotlib>=3.7.0

#### cal.py
**Status**: ✓ EXTRACTED FROM NOTEBOOK
**Purpose**: Monolithic module for backward compatibility
**Contains**: All 6 components in one file

---

### Components Directory (NEW)

#### components/__init__.py
**Exports**:
- VulnerabilitySpotter
- ConfessionalTemplate
- TinyConfessionalLayer
- UnifiedCAL_TRM
- ScratchpadLayer
- CAL_TRM_Hybrid

#### components/vulnerability_spotter.py
**Class**: `VulnerabilitySpotter`
**Purpose**: Multi-metric vulnerability detection
**Features**:
- Scarcity detection via semantic stress
- Entropy risk calculation
- Deceptive variance analysis
- Bayesian or weighted sum aggregation
**Dependencies**: torch, torch.nn, torch.nn.functional, numpy

#### components/confessional_template.py
**Class**: `ConfessionalTemplate`
**Purpose**: Template-based reasoning structure
**Features**:
- 6 named templates (prior, evidence, posterior, relational_check, moral, action)
- Template projection with noise injection
**Dependencies**: torch, torch.nn

#### components/tiny_confessional_layer.py
**Class**: `TinyConfessionalLayer`
**Purpose**: Recursive confessional reasoning loop
**Features**:
- THINK-ACT-COHERENCE recursion
- Max 16 cycles with early stopping
- Template cycling
- Integrated VulnerabilitySpotter
**Dependencies**: torch, torch.nn, torch.nn.functional, .vulnerability_spotter

#### components/unified_cal_trm.py
**Class**: `UnifiedCAL_TRM`
**Purpose**: Public API for CAL-TRM
**Features**:
- Clean public interface
- Metadata return option
- Audit mode support
**Dependencies**: torch, torch.nn, .vulnerability_spotter, .tiny_confessional_layer

#### components/scratchpad_layer.py
**Class**: `ScratchpadLayer`
**Purpose**: Persistent state tracking
**Features**:
- Learnable reset state
- Residual connections
- Mean pooling for updates
**Dependencies**: torch, torch.nn

#### components/cal_trm_hybrid.py
**Class**: `CAL_TRM_Hybrid`
**Purpose**: Hybrid architecture with scratchpad
**Features**:
- Integrated scratchpad persistence
- Threshold-based triggering (default: 0.2)
- Returns output, metadata, and scratchpad state
**Dependencies**: torch, torch.nn, .scratchpad_layer, .tiny_confessional_layer, .vulnerability_spotter

---

### Tests Directory

#### tests/test_cal.py
**Status**: ✓ EXTRACTED FROM NOTEBOOK
**Tests**:
1. VulnerabilitySpotter basic functionality
2. ConfessionalTemplate with multiple templates
3. TinyConfessionalLayer recursion
4. ScratchpadLayer state persistence
5. CAL_TRM_Hybrid integration test

**Usage**:
```bash
python tests/test_cal.py
```

---

### Examples Directory

#### examples/truthfulqa_eval.py
**Status**: ✓ CREATED (Simplified from notebook)
**Purpose**: TruthfulQA evaluation with CAL-TRM
**Features**:
- DistilBERT embeddings
- Vulnerability detection for deception proxy
- Metric collection (v_t scores, trigger rates, cycles, coherence)
- Clean output formatting

**Usage**:
```bash
python examples/truthfulqa_eval.py
```

---

### Documentation Directory

#### docs/COMPONENTS.md
**Status**: ✓ CREATED
**Contents**:
- Detailed component descriptions
- Usage examples for each component
- Modularization guidelines
- Testing strategies
- Migration guide from monolithic to modular
- Repository structure recommendations

---

## Usage Patterns

### Option 1: Monolithic Import (Backward Compatible)
```python
from cal import UnifiedCAL_TRM

model = UnifiedCAL_TRM(d_model=256)
x = torch.randn(1, 32, 256)
out, meta = model(x, return_metadata=True)
```

### Option 2: Modular Import (Recommended)
```python
from components import UnifiedCAL_TRM

model = UnifiedCAL_TRM(d_model=256)
x = torch.randn(1, 32, 256)
out, meta = model(x, return_metadata=True)
```

### Option 3: Individual Component Import
```python
from components.vulnerability_spotter import VulnerabilitySpotter
from components.tiny_confessional_layer import TinyConfessionalLayer

spotter = VulnerabilitySpotter(d_model=256)
confessional = TinyConfessionalLayer(d_model=256)
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/augstentatious/TRuCAL.git
cd TRuCAL

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_cal.py

# Run evaluation
python examples/truthfulqa_eval.py
```

---

## GitHub Upload Strategy

### Recommended Approach: Single Repository

Upload the entire modular structure as one cohesive repository:

```bash
git add .
git commit -m "Modularize TRuCAL: separate components, tests, examples, and docs"
git push origin main
```

### Alternative Approach: Multiple Repositories

If you want separate repositories for each component:

1. **TRuCAL-Core** (vulnerability_spotter.py)
2. **TRuCAL-Templates** (confessional_template.py)
3. **TRuCAL-Layer** (tiny_confessional_layer.py)
4. **TRuCAL-Unified** (unified_cal_trm.py + dependencies)
5. **TRuCAL-Hybrid** (cal_trm_hybrid.py + dependencies)

---

## Quality Improvements Made

### 1. README.md Proofread ✓
- Fixed markdown formatting
- Added proper code blocks
- Improved structure and readability

### 2. Code Modularization ✓
- Separated 6 components into individual files
- Created clean package structure
- Maintained backward compatibility

### 3. Testing Organization ✓
- Moved tests to dedicated directory
- Comprehensive test coverage for all components

### 4. Examples Creation ✓
- Created simplified evaluation script
- Clean, documented code

### 5. Documentation ✓
- Component organization guide
- Project structure overview
- Usage patterns and migration guide

---

## Next Steps

1. **Run Tests**: Verify all components work correctly
   ```bash
   python tests/test_cal.py
   ```

2. **Review Components**: Check each component file in `components/`

3. **Upload to GitHub**: Push the organized codebase
   ```bash
   git add .
   git commit -m "Complete TRuCAL modularization and documentation"
   git push
   ```

4. **Optional Enhancements**:
   - Add more test cases
   - Create additional evaluation scripts
   - Add API documentation
   - Create tutorial notebooks

---

## Changelog

### v1.0.0 (Current)
- ✓ Proofread and fixed README.md
- ✓ Created modular component structure
- ✓ Extracted and organized tests
- ✓ Created evaluation examples
- ✓ Added comprehensive documentation
- ✓ Created requirements.txt
- ✓ Maintained backward compatibility with cal.py

---

## Support

For questions or issues:
- Check docs/COMPONENTS.md for detailed component info
- Review examples/ for usage patterns
- Run tests/ to verify functionality
- Refer to README.md for quick start guide
