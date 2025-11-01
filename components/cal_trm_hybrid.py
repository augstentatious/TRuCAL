"""
CAL_TRM_Hybrid Module

Hybrid architecture combining scratchpad state persistence, vulnerability detection,
and confessional reasoning with threshold-based triggering.
"""

import torch
import torch.nn as nn
from .scratchpad_layer import ScratchpadLayer
from .tiny_confessional_layer import TinyConfessionalLayer
from .vulnerability_spotter import VulnerabilitySpotter


class CAL_TRM_Hybrid(nn.Module):
    """
    Hybrid CAL-TRM combining scratchpad, vulnerability detection, and confessional reasoning.
    """
    def __init__(self, d_model=256, confessional_threshold=0.2):
        super().__init__()
        self.scratchpad = ScratchpadLayer(d_model)
        self.cal_confessional = TinyConfessionalLayer(d_model)
        self.vuln_spotter = VulnerabilitySpotter(d_model)
        self.threshold = confessional_threshold

    def forward(self, x, prev_z=None, attention_weights=None, **kwargs):
        """
        Forward pass of CAL_TRM_Hybrid.

        Args:
            x: Input tensor (batch_size, sequence_length, d_model)
            prev_z: Previous scratchpad state (batch_size, d_model)
            attention_weights: Optional attention weights for vulnerability detection
            **kwargs: Additional arguments (e.g., audit_mode)

        Returns:
            output: Model output tensor
            metadata: Dictionary containing confessional metadata
            z_scratch: Updated scratchpad state
        """
        z_scratch = self.scratchpad(x, prev_z=prev_z)

        v_t, vs_metadata = self.vuln_spotter(x, attention_weights=attention_weights)

        v_t_trigger = torch.mean(v_t, dim=1).squeeze(-1)

        confessional_triggered = (v_t_trigger > self.threshold).any().item()

        if confessional_triggered:
            confession_out, cal_metadata = self.cal_confessional(x, attention_weights=attention_weights, audit_mode=kwargs.get('audit_mode', False))

            metadata = {
                'confessional_triggered': True,
                'v_t_trigger_values': v_t_trigger.detach().cpu().numpy(),
                'scratchpad_state': z_scratch.clone().detach().cpu().numpy(),
                'cal_metadata': cal_metadata
            }
            return confession_out, metadata, z_scratch
        else:
            metadata = {
                'confessional_triggered': False,
                'v_t_trigger_values': v_t_trigger.detach().cpu().numpy(),
                'scratchpad_state': z_scratch.clone().detach().cpu().numpy(),
                'cal_metadata': {}
            }
            return x, metadata, z_scratch
