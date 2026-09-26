"""Mojo port of agate's column statistics.

[agate](https://agate.readthedocs.io/) is a table library: CSV and JSON
readers, table plumbing, and a set of column aggregations. Only the
aggregations do arithmetic, so that is what is ported: min, max, sum, count,
mean, sample and population variance and standard deviation, median, agate's
101 percentiles, the quartile/quintile/decile views, IQR and MAD.

```python
import agate
import mojo_agate

table = agate.Table.from_csv(
    __import__("io").StringIO("n\\n1\\n2\\n3\\n4\\n"), column_types=["number"]
)
mojo_agate.stats(table, "n")["mean"]   # 2.5
```

Everything else in agate - readers and writers, pivot, join, group_by,
order_by, bins, formatting, the boolean/percentage/date data types - is left to
the real package.
"""

from .aggregations import (
    count,
    deciles,
    iqr,
    mad,
    max_,
    mean,
    median,
    min_,
    percentiles,
    population_stdev,
    population_variance,
    quintiles,
    quartiles,
    sorted_values,
    stats,
    stdev,
    sum_,
    values_of,
    variance,
)

__all__ = [
    "count",
    "deciles",
    "iqr",
    "mad",
    "max_",
    "mean",
    "median",
    "min_",
    "percentiles",
    "population_stdev",
    "population_variance",
    "quintiles",
    "quartiles",
    "sorted_values",
    "stats",
    "stdev",
    "sum_",
    "values_of",
    "variance",
]
__version__ = "0.1.0"
