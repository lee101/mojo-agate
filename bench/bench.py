"""Correctness-gated benchmark for mojo-agate.

Every case checks the Mojo result against the real agate aggregation before
timing, so a wrong kernel shows up as a correctness failure rather than as a
suspiciously good number.

The baselines are agate's own aggregations run on a one-column table: the
reductions, the sorts, the percentiles and the MAD. That is the fairest
comparison available, since agate is the thing being ported. Building the
agate table is outside the timed region for both sides.
"""

from __future__ import annotations

import pathlib
import sys
import time
from decimal import Decimal

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import agate  # noqa: E402
import mojo_agate  # noqa: E402
from agate.aggregations import (  # noqa: E402
    Count,
    MAD,
    Max,
    Mean,
    Median,
    Min,
    Percentiles,
    StDev,
    Sum,
    Variance,
)

N = 1 << 20


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _data(n=N, seed=0, nulls=0):
    rng = np.random.default_rng(seed)
    values = np.round(rng.uniform(0, 1000, n), 2)
    rows = []
    for i, v in enumerate(values):
        rows.append([None if i < nulls else Decimal(repr(float(v)))])
    return values, rows


def _table(rows):
    return agate.Table(rows, ["n"], [agate.Number()])


def bench_reduction(name, mine, theirs, n=N, nulls=0):
    values, rows = _data(n, nulls=nulls)
    table = _table(rows)
    expect = theirs("n").run(table)
    got = mine(values)
    if expect is None:
        assert np.isnan(got)
    else:
        np.testing.assert_allclose(got, float(expect), rtol=1e-11, atol=1e-11)
    agate_time = _time(lambda: theirs("n").run(table))
    mojo_time = _time(lambda: mine(values))
    return f"{name} n={n}", agate_time, mojo_time


def bench_percentiles(n=N):
    values, rows = _data(n)
    table = _table(rows)
    got = mojo_agate.percentiles(values)
    theirs = Percentiles("n").run(table)
    for p in range(101):
        np.testing.assert_allclose(got[p], float(theirs[p]), rtol=1e-11, atol=1e-11)
    agate_time = _time(lambda: Percentiles("n").run(table))
    mojo_time = _time(lambda: mojo_agate.percentiles(values))
    return f"percentiles(101) n={n}", agate_time, mojo_time


def bench_median(n=N):
    values, rows = _data(n, seed=1)
    table = _table(rows)
    np.testing.assert_allclose(
        mojo_agate.median(values), float(Median("n").run(table)), rtol=1e-11
    )
    agate_time = _time(lambda: Median("n").run(table))
    mojo_time = _time(lambda: mojo_agate.median(values))
    return f"median n={n}", agate_time, mojo_time


def bench_mad(n=N):
    values, rows = _data(n, seed=2)
    table = _table(rows)
    np.testing.assert_allclose(
        mojo_agate.mad(values), float(MAD("n").run(table)), rtol=1e-11
    )
    agate_time = _time(lambda: MAD("n").run(table))
    mojo_time = _time(lambda: mojo_agate.mad(values))
    return f"mad n={n}", agate_time, mojo_time


def bench_iqr(n=N):
    values, rows = _data(n, seed=3)
    table = _table(rows)
    np.testing.assert_allclose(
        mojo_agate.iqr(values), float(agate.aggregations.IQR("n").run(table)),
        rtol=1e-11,
    )
    agate_time = _time(lambda: agate.aggregations.IQR("n").run(table))
    mojo_time = _time(lambda: mojo_agate.iqr(values))
    return f"iqr n={n}", agate_time, mojo_time


def numpy_percentiles(values):
    """agate's CDF percentiles, expressed in NumPy after one sort."""
    ordered = np.sort(values)
    n = ordered.size
    out = np.empty(101, dtype=np.float64)
    out[0] = ordered[0]
    for p in range(1, 100):
        k = n * (p / 100.0)
        low = max(1, int(np.ceil(k)))
        high = min(n, int(np.floor(k + 1.0)))
        if low == high:
            out[p] = ordered[low - 1]
        else:
            out[p] = (ordered[low - 1] + ordered[high - 1]) / 2.0
    out[100] = ordered[-1]
    return out


def numpy_median(values):
    ordered = np.sort(values)
    n = ordered.size
    if n % 2 == 1:
        return float(ordered[(n + 1) // 2 - 1])
    return float((ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)


def numpy_mad(values):
    # agate's rule: deviations of the sorted column, midpoint of the third
    # and fourth for an even length
    ordered = np.sort(values)
    n = ordered.size
    centre = numpy_median(values)
    if n % 2 == 1:
        return abs(float(ordered[(n + 1) // 2 - 1]) - centre)
    half = n // 2
    return (
        abs(float(ordered[half - 1]) - centre) + abs(float(ordered[half]) - centre)
    ) / 2.0


def numpy_table():
    """The same cases against NumPy: a float64-against-float64 comparison."""
    print(f"{'case':<24}{'NumPy float64':>16}{'mojo-agate':>14}{'ratio':>9}")
    print("-" * 63)
    rng = np.random.default_rng(99)
    v = np.round(rng.uniform(0, 1000, N), 2)
    cases = [
        ("min", lambda: np.nanmin(v), lambda: mojo_agate.min_(v)),
        ("mean", lambda: np.nanmean(v), lambda: mojo_agate.mean(v)),
        ("variance", lambda: np.nanvar(v, ddof=1), lambda: mojo_agate.variance(v)),
        ("stdev", lambda: np.nanstd(v, ddof=1), lambda: mojo_agate.stdev(v)),
        ("median", lambda: numpy_median(v), lambda: mojo_agate.median(v)),
        ("percentiles(101)", lambda: numpy_percentiles(v),
         lambda: mojo_agate.percentiles(v)),
        ("mad", lambda: numpy_mad(v), lambda: mojo_agate.mad(v)),
    ]
    for label, reference, mine in cases:
        np.testing.assert_allclose(mine(), reference(), rtol=1e-11, atol=1e-11)
        ref_t = _time(reference)
        got_t = _time(mine)
        ratio = ref_t / got_t if got_t else float("nan")
        print(f"{label:<24}{ref_t*1e3:>14.2f}ms{got_t*1e3:>12.2f}ms{ratio:>8.2f}x")
    print()


def main():
    print("agate computes in arbitrary-precision Decimal; that ratio measures "
          "the arithmetic type as much as the language.\n")
    print(f"{'case':<24}{'agate':>14}{'mojo-agate':>14}{'ratio':>9}")
    print("-" * 61)
    cases = [
        lambda: bench_reduction("min", mojo_agate.min_, Min),
        lambda: bench_reduction("max", mojo_agate.max_, Max),
        lambda: bench_reduction("sum", mojo_agate.sum_, Sum),
        lambda: bench_reduction("count", mojo_agate.count, Count),
        lambda: bench_reduction("mean", mojo_agate.mean, Mean),
        lambda: bench_reduction("variance", mojo_agate.variance, Variance),
        lambda: bench_reduction("stdev", mojo_agate.stdev, StDev),
        lambda: bench_reduction("min (with nulls)", mojo_agate.min_, Min,
                                nulls=1000),
        lambda: bench_median(),
        lambda: bench_percentiles(),
        lambda: bench_iqr(),
        lambda: bench_mad(),
    ]
    for case in cases:
        label, ref, got = case()
        ratio = ref / got if got else float("nan")
        print(f"{label:<24}{ref*1e3:>12.2f}ms{got*1e3:>12.2f}ms{ratio:>8.2f}x")
    print()
    numpy_table()


if __name__ == "__main__":
    main()
