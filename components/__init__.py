"""
TRuCAL Components - Modular CAL-TRM Implementation
"""

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

__version__ = '1.0.0'
