"""
ScratchpadLayer Module

Persistent state tracking layer for multi-turn confessional reasoning.
Maintains a learnable scratchpad state that accumulates across reasoning steps.
"""

import torch
import torch.nn as nn


class ScratchpadLayer(nn.Module):
    """
    Scratchpad layer for maintaining persistent state across confessional reasoning cycles.
    """
    def __init__(self, d_model):
        super().__init__()
        self.pad_proj = nn.Linear(d_model, d_model)
        self.reset = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, x, prev_z=None):
        """
        Update scratchpad state with new input.

        Args:
            x: Input tensor (batch_size, sequence_length, d_model)
            prev_z: Previous scratchpad state (batch_size, d_model), None for reset

        Returns:
            Updated scratchpad state (batch_size, d_model)
        """
        if prev_z is None:
            prev_z = self.reset.expand(x.size(0), -1)
        x_pooled = x.mean(dim=1)
        z = self.pad_proj(x_pooled) + 0.7 * prev_z
        return z
