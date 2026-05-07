"""Utility functions for InterFormer."""

import torch
import numpy as np
from typing import List, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> str:
    """Get available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_shape=None):
    """Print model summary."""
    print(model)
    print(f"Total parameters: {count_parameters(model):,}")
    if input_shape:
        try:
            # Create a dummy input
            dummy_input = []
            for shape in input_shape:
                dummy_input.append(torch.randn(*shape))
            # Forward pass
            output = model(*dummy_input)
            print(f"Input shapes: {[s.shape for s in dummy_input]}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error generating summary: {e}")