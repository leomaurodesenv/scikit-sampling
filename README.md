# 🧪 Scikit-Sampling

[![GitHub](https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue&style=flat-square)](https://github.com/leomaurodesenv/scikit-sampling)
[![MIT license](https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square)](LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/leomaurodesenv/scikit-sampling/deployment.yml?label=Build&style=flat-square)](https://github.com/leomaurodesenv/scikit-sampling/actions/workflows/deployment.yml)


Scikit-Sampling (or `sksampling`) is a Python library for dataset sampling techniques. It provides a unified API for common sampling strategies, making it easy to integrate into your data science and machine learning workflows.

## Installation

You can install `sksampling` using pip:

```bash
pip install scikit-sampling
```

## Features

`sksampling` offers a range of sampling methods, including:

- `sample_size`: Computes the ideal sample size based confidence level and interval.

## Weighted Sampling

The `WeightedSampler` class allows you to sample data points with probabilities proportional to user-specified weights. This is useful for handling imbalanced datasets or prioritizing certain observations.

### Example Usage

```python
from sksampling import WeightedSampler

data = ['a', 'b', 'c', 'd']
weights = [0.1, 0.2, 0.6, 0.1]
sampler = WeightedSampler(data, weights)
sample = sampler.sample(n=3, replace=True, random_state=42)
print(sample)
```

- `data`: Sequence of items to sample from.
- `weights`: Sequence of non-negative weights (same length as data).
- `n`: Number of samples to draw.
- `replace`: Whether to sample with replacement (default: True).
- `random_state`: Optional seed for reproducibility.

The probability of each item being selected is proportional to its weight.

## Usage

`sksampling` follows the `