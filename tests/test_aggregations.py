"""Parity tests for the Mojo column statistics against the real agate.

Each test builds a real `agate.Table` and compares the Mojo aggregation with
the upstream `agate.aggregations` one on the same column. agate computes in
`decimal.Decimal` and the kernels in float64, so the comparison is a relative
tolerance, not equality; the data is chosen to be exactly representable in
float64 so the tolerance is never doing any work.

The cases are chosen to catch the plausible bugs: a rank off by one in
agate's percentile rule, a population/sample mix-up, a null that is reduced
instead of skipped, a median that takes the wrong element of an even-length
column, and a sort that is not stable.
"""

import numpy as np
import pytest

import agate
import mojo_agate
from agate.aggregations import (
    Count,
    Deciles,
    IQR,
    MAD,
    Max,
    Mean,
    Median,
    Min,
    Percentiles,
    PopulationStDev,
    PopulationVariance,
    Quintiles,
    Quartiles,
    StDev,
    Sum,
    Variance,
)

TOL = dict(rtol=1e-12, atol=1e-12)


def table(values, name="n"):
    return agate.Table([[v] for v in values], [name], [agate.Number()])


def check(mine, theirs, **tol):
    if theirs is None:
        assert mine is None or np.isnan(mine)
        return
    np.testing.assert_allclose(mine, float(theirs), **(tol or TOL))


def ref_percentiles(t, column="n"):
    return Percentiles(column).run(t)


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1],
        [-5, -1, 0, 1, 7, 100],
        [0.25, 0.5, 0.75, 2.5, 3.125],
        list(range(1, 101)),
        [10, 10, 20, 20, 30],
        [1, 1000000],
        [2, 2, 2, 3],
    ],
)
def test_scalar_aggregations_match_agate(values):
    t = table(values)
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.min_(mine), Min("n").run(t))
    check(mojo_agate.max_(mine), Max("n").run(t))
    check(mojo_agate.sum_(mine), Sum("n").run(t))
    check(mojo_agate.mean(mine), Mean("n").run(t))
    check(mojo_agate.count(mine), Count("n").run(t))
    check(mojo_agate.variance(mine), Variance("n").run(t))
    check(mojo_agate.population_variance(mine), PopulationVariance("n").run(t))
    check(mojo_agate.stdev(mine), StDev("n").run(t))
    check(mojo_agate.population_stdev(mine), PopulationStDev("n").run(t))
    check(mojo_agate.median(mine), Median("n").run(t))
    check(mojo_agate.iqr(mine), IQR("n").run(t))
    check(mojo_agate.mad(mine), MAD("n").run(t))


def test_percentiles_match_agate():
    values = list(range(1, 41)) + [7, 7, 7, 19, 23, 41]
    t = table(values)
    mine = mojo_agate.percentiles(mojo_agate.values_of(t, "n"))
    theirs = ref_percentiles(t)
    assert len(mine) == 101
    for p in range(101):
        check(mine[p], theirs[p], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100, 101, 250])
def test_percentile_ranks_match_agate_for_every_length(n):
    # agate's rule is k = n * p/100 with a ceil/floor pair, so every length
    # exercises a different combination of low == high and the averaged case
    rng = np.random.default_rng(1000 + n)
    values = sorted(np.round(rng.uniform(-500, 500, n), 2).tolist())
    t = table(values)
    mine = mojo_agate.percentiles(mojo_agate.values_of(t, "n"))
    theirs = ref_percentiles(t)
    for p in range(101):
        check(mine[p], theirs[p], rtol=1e-12, atol=1e-12)


def test_quartiles_quintiles_deciles_match_agate():
    values = [float(v) for v in range(1, 52)]
    t = table(values)
    mine = mojo_agate.values_of(t, "n")
    assert len(mojo_agate.quartiles(mine)) == 5
    assert mojo_agate.quartiles(mine) == pytest.approx(
        [float(v) for v in Quartiles("n").run(t)], rel=1e-12
    )
    assert len(mojo_agate.quintiles(mine)) == 6
    assert mojo_agate.quintiles(mine) == pytest.approx(
        [float(v) for v in Quintiles("n").run(t)], rel=1e-12
    )
    assert len(mojo_agate.deciles(mine)) == 11
    assert mojo_agate.deciles(mine) == pytest.approx(
        [float(v) for v in Deciles("n").run(t)], rel=1e-12
    )


def test_nulls_are_skipped_exactly_as_agate_does():
    values = [1, None, 3, 4, None, 10, 2, None, 8, 5, 5, 11]
    t = table(values)
    mine = mojo_agate.values_of(t, "n")
    # the kernel must reduce the nine non-null values, not ten or twelve
    assert mojo_agate.count(mine) == Count("n").run(t) == 9
    check(mojo_agate.min_(mine), Min("n").run(t))
    check(mojo_agate.max_(mine), Max("n").run(t))
    check(mojo_agate.mean(mine), Mean("n").run(t))
    check(mojo_agate.median(mine), Median("n").run(t))
    check(mojo_agate.mad(mine), MAD("n").run(t))
    ordered = mojo_agate.sorted_values(mine)
    assert ordered.tolist() == [
        float(v) for v in t.columns["n"].values_without_nulls_sorted()
    ]


def test_leading_and_trailing_nulls_do_not_shift_percentiles():
    values = [None, None] + [float(v) for v in range(1, 12)] + [None]
    t = table(values)
    mine = mojo_agate.values_of(t, "n")
    theirs = ref_percentiles(t)
    got = mojo_agate.percentiles(mine)
    for p in range(101):
        check(got[p], theirs[p], rtol=1e-12, atol=1e-12)


def test_all_null_column_matches_agate():
    t = table([None, None, None])
    mine = mojo_agate.values_of(t, "n")
    assert np.isnan(mojo_agate.min_(mine)) and Min("n").run(t) is None
    assert np.isnan(mojo_agate.mean(mine)) and Mean("n").run(t) is None
    assert np.isnan(mojo_agate.median(mine)) and Median("n").run(t) is None
    assert mojo_agate.count(mine) == Count("n").run(t) == 0
    assert all(np.isnan(v) for v in mojo_agate.percentiles(mine))
    assert all(ref_percentiles(t)[p] is None for p in range(101))


def test_single_value_column():
    t = table([42.5])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.min_(mine), Min("n").run(t))
    check(mojo_agate.mean(mine), Mean("n").run(t))
    check(mojo_agate.median(mine), Median("n").run(t))
    # sample variance of one value divides by zero in agate's formula
    assert np.isnan(mojo_agate.variance(mine))
    assert mojo_agate.population_variance(mine) == 0.0


def test_median_picks_the_right_element_of_an_even_column():
    # agate's utils.median: odd takes the middle, even averages the two
    # middles. An off-by-one here returns the upper or lower neighbour.
    t = table([1, 2, 3, 4])
    mine = mojo_agate.values_of(t, "n")
    assert mojo_agate.median(mine) == float(Median("n").run(t)) == 2.5
    t = table([1, 2, 3])
    mine = mojo_agate.values_of(t, "n")
    assert mojo_agate.median(mine) == float(Median("n").run(t)) == 2.0


def test_mad_reproduces_agates_unsorted_deviation_rule():
    # agate takes abs(value - median) in *sorted column order* and feeds that
    # to utils.median without sorting it, so for [1, 2, 3, 4, 5, 11] the
    # median is 3.5 and it averages the deviations of 3 and 4 (0.5 and 0.5)
    # rather than the two middle deviations (1.5 and 1.5). The textbook MAD
    # would be 1.5; a port that "fixed" this would not match agate.
    t = table([1, 2, 3, 4, 5, 11])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.mad(mine), MAD("n").run(t))
    assert mojo_agate.mad(mine) == 0.5
    # odd length takes the deviation of the middle element: the median of
    # [1, 2, 3, 4, 5, 20, 30] is 4, so the answer is |4 - 4| = 0
    t = table([1, 2, 3, 4, 5, 20, 30])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.mad(mine), MAD("n").run(t))
    assert mojo_agate.median(mine) == 4.0
    assert mojo_agate.mad(mine) == 0.0
    # a non-zero odd-length case: the median is 9 and it is itself a member,
    # so the deviation of the middle element is zero
    t = table([1, 2, 8, 9, 15, 20, 30])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.mad(mine), MAD("n").run(t))
    assert mojo_agate.mad(mine) == 0.0
    t = table([1, 2, 8, 9, 15, 20, 30, 40])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.mad(mine), MAD("n").run(t))
    assert mojo_agate.median(mine) == 12.0
    assert mojo_agate.mad(mine) == 3.0


def test_sorted_values_is_ascending_and_complete():
    rng = np.random.default_rng(3)
    values = rng.uniform(-1e3, 1e3, 500)
    ordered = mojo_agate.sorted_values(values)
    assert ordered.shape == (500,)
    assert np.all(np.diff(ordered) >= 0)
    np.testing.assert_array_equal(ordered, np.sort(values))


def test_sort_handles_sizes_around_the_merge_widths():
    # the merge sort ping-pongs between two buffers, so sizes either side of
    # a power of two are where an off-by-one in the pass bookkeeping shows up
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 129):
        rng = np.random.default_rng(n)
        values = rng.standard_normal(n)
        ordered = mojo_agate.sorted_values(values)
        assert ordered.shape == (n,)
        np.testing.assert_array_equal(ordered, np.sort(values))


def test_stats_dict_matches_every_agate_aggregation():
    values = [float(v) for v in range(1, 30)] + [None, 4.5, 4.5]
    t = table(values)
    got = mojo_agate.stats(t, "n")
    assert got["count"] == Count("n").run(t)
    check(got["min"], Min("n").run(t))
    check(got["max"], Max("n").run(t))
    check(got["sum"], Sum("n").run(t))
    check(got["mean"], Mean("n").run(t))
    check(got["variance"], Variance("n").run(t))
    check(got["population_variance"], PopulationVariance("n").run(t))
    check(got["stdev"], StDev("n").run(t))
    check(got["population_stdev"], PopulationStDev("n").run(t))
    check(got["median"], Median("n").run(t))
    check(got["iqr"], IQR("n").run(t))
    check(got["mad"], MAD("n").run(t))
    assert got["median"] == got["percentiles"][50]


def test_column_is_not_mutated():
    values = np.array([3.0, 1.0, 2.0, np.nan, 5.0])
    before = values.copy()
    mojo_agate.sorted_values(values)
    mojo_agate.mad(values)
    mojo_agate.percentiles(values)
    np.testing.assert_array_equal(values, before)


def test_negative_and_zero_data():
    t = table([-9, -3, 0, 0, 3, 9])
    mine = mojo_agate.values_of(t, "n")
    check(mojo_agate.min_(mine), Min("n").run(t))
    check(mojo_agate.max_(mine), Max("n").run(t))
    check(mojo_agate.mean(mine), Mean("n").run(t))
    check(mojo_agate.variance(mine), Variance("n").run(t))
    check(mojo_agate.median(mine), Median("n").run(t))
    check(mojo_agate.mad(mine), MAD("n").run(t))
    theirs = ref_percentiles(t)
    got = mojo_agate.percentiles(mine)
    for p in range(101):
        check(got[p], theirs[p], rtol=1e-12, atol=1e-12)


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        mojo_agate.mean(np.zeros((2, 2)))
