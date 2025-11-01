"""
ConfessionalTemplate Module

Template module for structuring private confessional reasoning with named templates.
Inspired by St. Augustine's Confessions structure.
"""

import torch
import torch.nn as nn


class ConfessionalTemplate(nn.Module):
    """
    Confessional Template module for generating different confessional outputs
    using named templates.
    """
    TEMPLATES = ["prior", "evidence", "posterior", "relational_check", "moral", "action"]

    def __init__(self, d_model: int = 256):
        """
        Initializes the ConfessionalTemplate with named templates.

        Args:
            d_model: The dimensionality of the input and output features.
        """
        super().__init__()
        self.d_model = d_model
        self.template_proj = nn.ModuleDict({k: nn.Linear(d_model, d_model) for k in self.TEMPLATES})

    def structure_reasoning(self, z: torch.Tensor, step: str = 'prior') -> torch.Tensor:
        """
        Applies a template projection to the input tensor.

        Args:
            z: Input tensor representing the private thought state (batch_size, sequence_length, d_model).
            step: The name of the template to use (e.g., 'prior').
        """
        if step in self.template_proj:
            return self.template_proj[step](z) + torch.randn_like(z) * 0.01
        return z

    def forward(self, z: torch.Tensor, step: str = 'prior') -> torch.Tensor:
        """
        Forward pass of the ConfessionalTemplate.

        Args:
            z: Input tensor representing the private thought state (batch_size, sequence_length, d_model).
            step: The name of the template to use (e.g., 'prior').

        Returns:
            Output tensor after applying the selected template.
        """
        return self.structure_reasoning(z, step)
