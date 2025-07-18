import math
import scipy.stats as st
import numpy as np
from typing import Sequence, Optional


def _get_z_score(confidence_level: float) -> float:
    """
    Retrieves the Z-score for a given confidence level.

    Args:
        confidence_level: The confidence level as a float (e.g., 0.95 for 95%).

    Returns:
        The Z-score for the given confidence level.
    """
    return st.norm.ppf(1 - (1 - confidence_level) / 2)


def sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    confidence_interval: float = 0.02,
) -> int:
    """
    Calculates the sample size for a finite population using Cochran's formula.

    Args:
        population_size: The total size of the population.
        confidence_level: The desired confidence level (e.g., 0.95 for 95%).
        confidence_interval: The desired confidence interval (margin of error).
                             Default is 0.02 (2%).

    Returns:
        The calculated sample size as an integer.
    """

    # For sample size calculation, we assume the worst-case variance, where p=0.5
    p = 0.5
    z_score = _get_z_score(confidence_level)
    # Calculate sample size for an infinite population
    n_0 = (z_score**2 * p * (1 - p)) / (confidence_interval**2)
    # Adjust sample size for the finite population
    n = n_0 / (1 + (n_0 - 1) / population_size)

    return int(math.ceil(n))


class WeightedSampler:
    """
    WeightedSampler allows sampling data points with probabilities proportional to user-specified weights.

    Args:
        data: Sequence of data points to sample from.
        weights: Sequence of weights corresponding to each data point.
    """
    def __init__(self, data: Sequence, weights: Sequence[float]):
        if len(data) != len(weights):
            raise ValueError("Data and weights must be the same length.")
        self.data = np.array(data)
        self.weights = np.array(weights, dtype=float)
        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative.")
        total = self.weights.sum()
        if total == 0:
            raise ValueError("Sum of weights must be positive.")
        self.probabilities = self.weights / total

    def sample(self, n: int, replace: bool = True, random_state: Optional[int] = None):
        """
        Draws a weighted sample from the data.

        Args:
            n: Number of samples to draw.
            replace: Whether to sample with replacement (default: True).
            random_state: Optional random seed for reproducibility.
        Returns:
            A numpy array of sampled data points.
        """
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(self.data), size=n, replace=replace, p=self.probabilities)
        return self.data[indices]
