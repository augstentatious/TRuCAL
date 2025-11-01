# TRuCAL Component Organization

This document organizes the TRuCAL codebase into individual components for modular GitHub uploads.

## Component 1: VulnerabilitySpotter

**File**: `vulnerability_spotter.py`

**Description**: Multi-metric risk aggregation module that detects vulnerabilities using scarcity, entropy, and deceptive variance metrics.

**Key Features**:
- Bayesian log-odds or weighted sum aggregation
- Shannon entropy calculation
- Semantic stress detection
- Deceptive variance analysis

**Dependencies**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

**Usage**:
```python
from vulnerability_spotter import VulnerabilitySpotter

spotter = VulnerabilitySpotter(d_model=256, aggregation_method='bayesian')
v_t, metadata = spotter(x, attention_weights=None)
```

---

## Component 2: ConfessionalTemplate

**File**: `confessional_template.py`

**Description**: Template module for structuring private confessional reasoning with named templates.

**Key Features**:
- 6 named templates: prior, evidence, posterior, relational_check, moral, action
- Template projection with noise injection
- Structured reasoning support

**Dependencies**:
```python
import torch
import torch.nn as nn
```

**Usage**:
```python
from confessional_template import ConfessionalTemplate

template = ConfessionalTemplate(d_model=256)
output = template(z, step='prior')
```

---

## Component 3: TinyConfessionalLayer

**File**: `tiny_confessional_layer.py`

**Description**: Recursive think/act confessional loop with template cycling and early stopping via coherence.

**Key Features**:
- THINK-ACT-COHERENCE recursion loop
- Max 16 cycles with early stopping (coherence ≥ 0.85)
- Template cycling through 6 stages
- Integrated VulnerabilitySpotter
- KL divergence and cosine similarity for coherence

**Dependencies**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from vulnerability_spotter import VulnerabilitySpotter
```

**Usage**:
```python
from tiny_confessional_layer import TinyConfessionalLayer

layer = TinyConfessionalLayer(d_model=256, n_inner=6, max_cycles=16)
output, metadata = layer(x, attention_weights=None, audit_mode=False)
```

---

## Component 4: UnifiedCAL_TRM

**File**: `unified_cal_trm.py`

**Description**: Unified CAL-TRM module integrating VulnerabilitySpotter and TinyConfessionalLayer with public API.

**Key Features**:
- Clean public API
- Optional metadata return
- Audit mode support
- Redacts private z state

**Dependencies**:
```python
import torch
import torch.nn as nn
from vulnerability_spotter import VulnerabilitySpotter
from tiny_confessional_layer import TinyConfessionalLayer
```

**Usage**:
```python
from unified_cal_trm import UnifiedCAL_TRM

model = UnifiedCAL_TRM(d_model=256)
output, metadata = model(x, return_metadata=True)
```

---

## Component 5: ScratchpadLayer

**File**: `scratchpad_layer.py`

**Description**: Persistent state tracking layer for multi-turn confessional reasoning.

**Key Features**:
- Learnable reset state
- Residual connections
- Mean pooling for state updates

**Dependencies**:
```python
import torch
import torch.nn as nn
```

**Usage**:
```python
from scratchpad_layer import ScratchpadLayer

scratchpad = ScratchpadLayer(d_model=256)
z = scratchpad(x, prev_z=None)
```

---

## Component 6: CAL_TRM_Hybrid

**File**: `cal_trm_hybrid.py`

**Description**: Hybrid architecture combining scratchpad state, vulnerability detection, and confessional reasoning.

**Key Features**:
- Integrated scratchpad persistence
- Threshold-based triggering (default: 0.2)
- Returns output, metadata, and scratchpad state

**Dependencies**:
```python
import torch
import torch.nn as nn
from scratchpad_layer import ScratchpadLayer
from tiny_confessional_layer import TinyConfessionalLayer
from vulnerability_spotter import VulnerabilitySpotter
```

**Usage**:
```python
from cal_trm_hybrid import CAL_TRM_Hybrid

model = CAL_TRM_Hybrid(d_model=256, confessional_threshold=0.2)
output, metadata, z_scratch = model(x, prev_z=None)
```

---

## Component Structure for GitHub

### Repository Structure

```
TRuCAL/
├── README.md                      # Main project documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── cal.py                         # All-in-one module (current)
│
├── components/                    # Individual component modules
│   ├── __init__.py
│   ├── vulnerability_spotter.py
│   ├── confessional_template.py
│   ├── tiny_confessional_layer.py
│   ├── unified_cal_trm.py
│   ├── scratchpad_layer.py
│   └── cal_trm_hybrid.py
│
├── tests/                         # Testing modules
│   ├── test_cal.py
│   └── test_components.py
│
├── examples/                      # Example scripts
│   ├── truthfulqa_eval.py
│   └── basic_usage.py
│
└── docs/                          # Documentation
    ├── COMPONENTS.md              # This file
    ├── ARCHITECTURE.md
    └── API.md
```

---

## Modularization Steps

To separate the monolithic `cal.py` into individual components:

### Step 1: Create Component Files

Extract each class from `cal.py` into its own file in the `components/` directory.

### Step 2: Update Imports

Each component file should import only its dependencies:

**vulnerability_spotter.py**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

**tiny_confessional_layer.py**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vulnerability_spotter import VulnerabilitySpotter
```

### Step 3: Create __init__.py

```python
from .vulnerability_spotter import VulnerabilitySpotter
from .confessional_template import ConfessionalTemplate
from .tiny_confessional_layer import TinyConfessionalLayer
from .unified_cal_trm import UnifiedCAL_TRM
from .scratchpad_layer import ScratchpadLayer
from .cal_trm_hybrid import CAL_TRM_Hybrid

__all__ = [
    'VulnerabilitySpotter',
    'ConfessionalTemplate',
    'TinyConfessionalLayer',
    'UnifiedCAL_TRM',
    'ScratchpadLayer',
    'CAL_TRM_Hybrid',
]
```

### Step 4: Update Usage

```python
# Before (monolithic)
from cal import UnifiedCAL_TRM

# After (modular)
from components import UnifiedCAL_TRM
# or
from components.unified_cal_trm import UnifiedCAL_TRM
```

---

## Component Testing

Each component should have dedicated tests:

**tests/test_vulnerability_spotter.py**:
```python
import torch
from components import VulnerabilitySpotter

def test_vulnerability_spotter():
    d_model = 256
    batch_size = 4
    seq_len = 10
    
    spotter = VulnerabilitySpotter(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    v_t, metadata = spotter(x)
    
    assert v_t.shape == (batch_size, seq_len, 1)
    assert 'scarcity' in metadata
    assert 'entropy_risk' in metadata
    assert 'deceptive' in metadata
```

---

## Documentation for Each Component

Each component file should include:

1. **Docstring**: Module description
2. **Class docstring**: Detailed class functionality
3. **Method docstrings**: Parameter descriptions and return values
4. **Usage examples**: Code snippets showing typical usage
5. **References**: Links to relevant papers or documentation

---

## Benefits of Modular Structure

1. **Easier Testing**: Test components in isolation
2. **Better Maintainability**: Clear separation of concerns
3. **Reusability**: Import only what you need
4. **Extensibility**: Easy to add new components
5. **Documentation**: Clearer API documentation per component
6. **Version Control**: Track changes to specific components
7. **Collaboration**: Multiple developers can work on different components

---

## Migration Guide

For users of the current monolithic `cal.py`:

**Option 1: Keep using cal.py (backward compatible)**
```python
from cal import UnifiedCAL_TRM  # Still works
```

**Option 2: Switch to modular imports**
```python
from components import UnifiedCAL_TRM  # Recommended
```

Both options will be supported for backward compatibility.
