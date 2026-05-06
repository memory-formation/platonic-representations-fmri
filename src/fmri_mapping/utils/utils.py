"""
This code is sourced and adapted from the repository:
    https://github.com/rjha18/vec2vec

Please cite the original work as:

Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025).
*Harnessing the universal geometry of embeddings*.
arXiv preprint arXiv:2505.12540. https://arxiv.org/abs/2505.12540
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set the random seed for reproducibility
    across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def exit_on_nan(loss: torch.Tensor) -> None:
    """Exit the program if the loss contains NaN values."""
    if torch.isnan(loss).any():
        print("Loss is NaN! exiting")
        exit(1)


def get_device() -> torch.device:
    """Get the available device (GPU if available, else CPU).

    Returns:
        torch.device: The device to use for computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_item(tensor):
    return tensor.item() if isinstance(tensor, torch.Tensor) else tensor
