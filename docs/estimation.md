# Estimation Functions

The `scikit-sampling` library provides helper functions to assist with the statistical aspects of sampling, such as determining an appropriate sample size or understanding the confidence level of an existing sample.

These functions are based on Cochran's formula for finite populations.

## `sample_size`

This function calculates the required sample size for a given population to achieve a desired confidence level and margin of error. It's essential for planning surveys and experiments to ensure the results are statistically significant.

### Usage

```python
from sksampling.estimation import sample_size

# Calculate the sample size for a population of 100,000
# with a 95% confidence level and a 2% margin of error.
required_sample = sample_size(
    population_size=100_000,
    confidence_level=0.95,
    confidence_interval=0.02
)

print(f"Required sample size: {required_sample}")
# Expected output: Required sample size: 2345
```

### Parameters

-   `population_size` (int): The total size of the population you are sampling from.
-   `confidence_level` (float, optional): The desired confidence level for your sample. Defaults to `0.95` (95%).
-   `confidence_interval` (float, optional): The desired margin of error. Defaults to `0.02` (2%).

### Returns

-   (int): The calculated minimum sample size required, rounded up to the nearest integer.

## `confidence_level`

This function is the inverse of `sample_size`. Given a sample size, population size, and a margin of error, it calculates the confidence level you can have in that sample's representativeness. This is useful for evaluating the statistical power of an existing sample.

### Usage

```python
from sksampling.estimation import confidence_level

# Calculate the confidence level for a sample of 2345 from a
# population of 100,000 with a 2% margin of error.
conf_level = confidence_level(
    sample_size=2345,
    population_size=100_000,
    confidence_interval=0.02
)

print(f"Calculated confidence level: {conf_level:.2f}")
# Expected output: Calculated confidence level: 0.95
```

### Parameters

-   `sample_size` (int): The size of the sample you have.
-   `population_size` (int): The total size of the population from which the sample was drawn.
-   `confidence_interval` (float, optional): The margin of error. Defaults to `0.02` (2%).

### Returns

-   (float): The calculated confidence level (e.g., `0.95` for 95%).
