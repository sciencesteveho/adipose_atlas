"""Global seed utility."""

import os
import random

import numpy as np


def _set_global_seed(seed: int) -> None:
    """Set seed for deterministic analyses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
