"""agate's column aggregations, with the same names and the same answers.

Every function takes a 1-D float64 array in which NaN marks a null, which is
how agate's `values_without_nulls()` sees a column: nulls are dropped before
the reduction, not treated as a value. agate computes in `decimal.Decimal`,
these kernels in float64, so results agree to rounding rather than bit for
bit.

Where agate returns `None` (an all-null column) these return `nan`;
`stats()` converts that back to `None` to match upstream exactly.
"""

import numpy as np

from . import _lib

__all__ = [
    "min_",
    "max_",
    "sum_",
    "count",
    "mean",
    "variance",
    "population_variance",
    "stdev",
    "population_stdev",
    "sorted_values",
    "median",
    "percentiles",
    "deciles",
    "quartiles",
    "quintiles",
    "iqr",
    "mad",
    "stats",
    "values_of",
]


def _values(values):
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("agate columns are one-dimensional")
    return arr


def min_(values):
    """`agate.aggregations.Min`."""
    return _lib.reduce_one(_values(values), _lib.MIN)


def max_(values):
    """`agate.aggregations.Max`."""
    return _lib.reduce_one(_values(values), _lib.MAX)


def sum_(values):
    """`agate.aggregations.Sum`."""
    return _lib.reduce_one(_values(values), _lib.SUM)


def count(values):
    """`agate.aggregations.Count`: the number of non-null values."""
    return _lib.reduce_one(_values(values), _lib.COUNT)


def mean(values):
    """`agate.aggregations.Mean`."""
    return _lib.reduce_one(_values(values), _lib.MEAN)


def variance(values):
    """`agate.aggregations.Variance`: the sample variance, /(n - 1)."""
    return _lib.dispersion_one(_values(values), _lib.VARIANCE)


def population_variance(values):
    """`agate.aggregations.PopulationVariance`, /n."""
    return _lib.dispersion_one(_values(values), _lib.POP_VARIANCE)


def stdev(values):
    """`agate.aggregations.StDev`: sqrt of the sample variance."""
    return _lib.dispersion_one(_values(values), _lib.STDEV)


def population_stdev(values):
    """`agate.aggregations.PopulationStDev`."""
    return _lib.dispersion_one(_values(values), _lib.POP_STDEV)


def sorted_values(values):
    """`Column.values_without_nulls_sorted()`: drop the nulls and order them."""
    return _lib.sort_values(_values(values))[0]


def median(values):
    """`agate.aggregations.Median`, the 50th percentile."""
    ordered, count = _lib.sort_values(_values(values))
    return _lib.median_of_sorted(ordered) if count else float("nan")


def percentiles(values):
    """`agate.aggregations.Percentiles`: 101 values, 0th through 100th."""
    ordered, count = _lib.sort_values(_values(values))
    if not count:
        return [float("nan")] * 101
    return _lib.percentiles_of_sorted(ordered)


def deciles(values):
    """`agate.aggregations.Deciles`: the 0th, 10th ... 100th percentiles."""
    p = percentiles(values)
    return [p[i] for i in range(0, 101, 10)]


def quartiles(values):
    """`agate.aggregations.Quartiles`."""
    p = percentiles(values)
    return [p[i] for i in (0, 25, 50, 75, 100)]


def quintiles(values):
    """`agate.aggregations.Quintiles`."""
    p = percentiles(values)
    return [p[i] for i in (0, 20, 40, 60, 80, 100)]


def iqr(values):
    """`agate.aggregations.IQR`: the 75th percentile less the 25th."""
    p = percentiles(values)
    if np.isnan(p[75]) or np.isnan(p[25]):
        return float("nan")
    return p[75] - p[25]


def mad(values):
    """`agate.aggregations.MAD`: median absolute deviation."""
    ordered, count = _lib.sort_values(_values(values))
    if not count:
        return float("nan")
    centre = _lib.median_of_sorted(ordered)
    return _lib.mad(ordered, centre)


def _or_none(value):
    return None if isinstance(value, float) and np.isnan(value) else value


def values_of(table, column_name):
    """The float64 contents of an agate column, with nulls as NaN.

    agate stores Number columns as `Decimal`; this is the conversion the shim
    owns, so the kernels only ever see contiguous float64.
    """
    column = table.columns[column_name]
    raw = column.values
    if callable(raw):  # `Column.values` is a method in agate 1.14
        raw = raw()
    out = np.empty(len(raw), dtype=np.float64)
    for i, value in enumerate(raw):
        out[i] = np.nan if value is None else float(value)
    return out


def stats(table, column_name):
    """Every aggregation this port covers for one agate column, in one dict.

    Values are `None` where agate's aggregation returns `None`, so this dict
    can be compared key by key with the upstream aggregations.
    """
    values = values_of(table, column_name)
    p = percentiles(values)
    return {
        "min": _or_none(min_(values)),
        "max": _or_none(max_(values)),
        "sum": _or_none(sum_(values)),
        "count": count(values),
        "mean": _or_none(mean(values)),
        "variance": _or_none(variance(values)),
        "population_variance": _or_none(population_variance(values)),
        "stdev": _or_none(stdev(values)),
        "population_stdev": _or_none(population_stdev(values)),
        "median": _or_none(median(values)),
        "percentiles": [_or_none(v) for v in p],
        "iqr": _or_none(iqr(values)),
        "mad": _or_none(mad(values)),
    }
