from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Forecast:
    """A forecast whose values are in target space."""

    prediction: torch.Tensor
    sample_ids: tuple[str, ...]
    input_end_indices: np.ndarray
    target_fill_mask: np.ndarray
    variance: torch.Tensor | None = None
