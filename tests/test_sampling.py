import pytest
from sksampling import _get_z_score, sample_size


def test_sample_size_basic():
    """
    Tests the sample_size function with a few known sets of inputs and
    expected outputs.
    """
    error_margin = 1
    # Test case from original print statement
    assert sample_size(100_000, 0.95, 0.02) == pytest.approx(2345, abs=error_margin)
    # Test with a smaller population
    assert sample_size(500, 0.95, 0.05) == pytest.approx(218, abs=error_margin)
    # Test with higher confidence and smaller interval
    assert sample_size(10_000, 0.99, 0.01) == pytest.approx(6239, abs=error_margin)
    # Test with a very large population, approaching the infinite case
    assert sample_size(1_000_000, 0.95, 0.05) == pytest.approx(385, abs=error_margin)


def test_z_score():
    """
    Tests the _get_z_score helper function with common confidence levels.
    """
    error_margin = 1e-2
    # Z-score for 90% confidence level should be approximately 1.645
    assert _get_z_score(0.90) == pytest.approx(1.645, abs=error_margin)
    # Z-score for 95% confidence level should be approximately 1.96
    assert _get_z_score(0.95) == pytest.approx(1.96, abs=error_margin)
    # Z-score for 99% confidence level should be approximately 2.58
    assert _get_z_score(0.99) == pytest.approx(2.58, abs=error_margin)


def test_weighted_sampler_basic():
    from sksampling import WeightedSampler
    data = ['a', 'b', 'c']
    weights = [0.1, 0.8, 0.1]
    sampler = WeightedSampler(data, weights)
    sample = sampler.sample(n=1000, replace=True, random_state=123)
    # 'b' should appear much more frequently
    b_count = (sample == 'b').sum()
    assert b_count > 700


def test_weighted_sampler_equal_weights():
    from sksampling import WeightedSampler
    data = [1, 2, 3, 4]
    weights = [1, 1, 1, 1]
    sampler = WeightedSampler(data, weights)
    sample = sampler.sample(n=1000, replace=True, random_state=1)
    # All items should appear roughly equally
    counts = [ (sample == i).sum() for i in data ]
    for c in counts:
        assert 200 < c < 300


def test_weighted_sampler_zero_weights():
    from sksampling import WeightedSampler
    data = [1, 2, 3]
    weights = [1, 0, 0]
    sampler = WeightedSampler(data, weights)
    sample = sampler.sample(n=100, replace=True, random_state=0)
    assert all(x == 1 for x in sample)


def test_weighted_sampler_invalid_weights():
    from sksampling import WeightedSampler
    import pytest
    # Negative weights
    with pytest.raises(ValueError):
        WeightedSampler([1,2], [1,-1])
    # All zero weights
    with pytest.raises(ValueError):
        WeightedSampler([1,2], [0,0])
    # Length mismatch
    with pytest.raises(ValueError):
        WeightedSampler([1,2,3], [1,2])


def test_weighted_sampler_no_replace():
    from sksampling import WeightedSampler
    data = ['x', 'y', 'z']
    weights = [0.2, 0.5, 0.3]
    sampler = WeightedSampler(data, weights)
    sample = sampler.sample(n=3, replace=False, random_state=42)
    assert set(sample) <= set(data)
    assert len(set(sample)) == 3
