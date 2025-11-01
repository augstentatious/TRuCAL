"""
UnifiedCAL_TRM Module

Unified CAL-TRM module integrating VulnerabilitySpotter and TinyConfessionalLayer.
Provides a clean public API with metadata option and redacts private z state.
"""

import torch
import torch.nn as nn
from .vulnerability_spotter import VulnerabilitySpotter
from .tiny_confessional_layer import TinyConfessionalLayer


class UnifiedCAL_TRM(nn.Module):
    """
    Unified CAL-TRM module integrating VulnerabilitySpotter and TinyConfessionalLayer.
    """
    def __init__(self, d_model: int = 256):
        """
        Initializes the UnifiedCAL_TRM.

        Args:
            d_model: The dimensionality of the input and output features.
        """
        super().__init__()
        self.vulnerability_spotter = VulnerabilitySpotter(d_model)
        self.tiny_confessional_layer = TinyConfessionalLayer(d_model)

    def forward(self, x: torch.Tensor, attention_weights=None, return_metadata: bool = False, audit_mode: bool = False):
        """
        Forward pass of the UnifiedCAL_TRM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).
            attention_weights: Optional attention weights for entropy calculation.
            return_metadata: If True, returns a tuple of (output, metadata).
            audit_mode: If True, enables audit mode in TinyConfessionalLayer.

        Returns:
            Output tensor after processing through CAL-TRM modules, or
            a tuple of (output, metadata) if return_metadata is True.
        """
        if return_metadata:
            cal_trm_output, metadata = self.tiny_confessional_layer(x, attention_weights=attention_weights, audit_mode=audit_mode)
            return cal_trm_output, metadata
        else:
            cal_trm_output, _ = self.tiny_confessional_layer(x, attention_weights=attention_weights, audit_mode=audit_mode)
            return cal_trm_output
