# mojo-agate

Mojo port of the compute surface of [agate](https://agate.readthedocs.io/)
(the Python CSV toolkit), version 1.14.2. The Python package is named
`mojo_agate`, so it installs alongside the real `agate` and the parity tests
import both and compare them directly.

## What the compute core is

agate is a table library: 6,940 lines, most of it CSV and JSON readers and
writers, table plumbing (`pivot`, `join`, `group_by`, `order_by`, `bins`,
`normalize`), and formatting through `babel`. The part that does arithmetic is
`agate/aggregations/`, and every one of those aggregations is a reduction over
a single Number column. That is what is ported.

## Covered subset

| area | implemented API |
| --- | --- |
| single-pass reductions | `min_`, `max_`, `sum_`, `count`, `mean` |
| dispersion | `variance` (sample, n-1), `population_variance` (n), `stdev`, `population_stdev` |
| order statistics | `sorted_values` (`values_without_nulls_sorted`), `median` (`agate.utils.median`) |
| percentiles | `percentiles` (agate's CDF method, 0th through 100th), `quartiles`, `quintiles`, `deciles`, `iqr` |
| robust spread | `mad` |
| table glue | `values_of(table, column)`, `stats(table, column)` |
| kernels | `agate_reduce`, `agate_dispersion`, `agate_sort`, `agate_sort_values`, `agate_median`, `agate_percentiles`, `agate_mad` |

Semantics reproduced from upstream, each with a parity test:

- nulls are dropped before the reduction, never reduced as a value. In the
  buffer a null is NaN, which is the only thing that can express "missing" in
  a float64 array;
- an all-null column gives `None` for every aggregation except `Count`, which
  gives 0. `stats()` converts the kernels' NaN back to `None` so the dict can
  be compared with upstream's return values;
- `Variance` needs at least two values, since it divides by n-1;
- agate's percentile rank is reproduced literally: `k = n * (p / 100)`,
  `low = max(1, ceil(k))`, `high = min(n, floor(k + 1))`, and the answer is
  `data[low-1]` when they agree and the mean of `data[low-1]` and
  `data[high-1]` when they do not. The 0th percentile is the first datum and
  the 100th is the last, whatever the arithmetic would give;
- `median` uses agate's `utils.median` rule, so an even-length column averages
  the two middle elements;
- **`MAD` reproduces an upstream quirk.** agate computes
  `median(tuple(abs(n - m) for n in data))` where `data` is the *sorted*
  column, and feeds deviations that are not themselves sorted into a function
  documented as requiring sorted input. For an even-length column agate
  therefore averages the deviations of the two middle *elements* rather than
  the two middle deviations. The textbook median absolute deviation is a
  different number, and this port deliberately returns agate's.

agate computes in `decimal.Decimal`; these kernels compute in float64. Results
therefore agree to rounding, not bit for bit. The parity tests use data that
is exactly representable in float64 and assert `rtol=atol=1e-12`.

## Not implemented

- `bins`, because its bin edges come from `utils.round_limits`, which is
  significant-digit arithmetic on `Decimal.normalize().as_tuple()` and has no
  float64 meaning.
- `order_by` and the other row-level sorts: agate sorts Python objects with
  its `NullOrder` sentinel, so there is no numeric kernel to extract, and a
  sort that is not stable on the caller's key object would silently change the
  answer.
- `pivot`, `join`, `group_by`, `distinct`, `denormalize`, `compute` and the
  rest of the table plumbing.
- `Mode`, `MaxLength`, `MaxPrecision`, `All`, `Any`, `HasNulls`, `First`,
  `Summary`: these are either string or boolean scans, or need a value-to-count
  map rather than a reduction.
- The Date, DateTime, TimeDelta, Text, Boolean and Percentage data types and
  all of `babel`'s number/date rendering.
- Every reader and writer (CSV, JSON, fixed-width).

## Install

```bash
pixi install
pixi run build     # -> dist/libmojo-agate.so
pixi run test
pixi run bench
```

## Tests

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

37 parity tests against the real `agate.aggregations`, on columns that are
ascending, descending, constant, negative, null-bearing, all-null, single
valued, two valued, and of every length from 1 to 250 (so every combination of
agate's `ceil`/`floor` rank branches is hit). They also cover the merge sort
either side of each of its power-of-two widths, input immutability, and the
even/odd median rule with hand-computed answers.

## Performance

Best-of-three, same process, n = 1,048,576 values, every case verified against
agate before timing.

agate computes in arbitrary-precision `Decimal`, so the first table mostly
measures the arithmetic type rather than the language. It is the honest
comparison against the package being ported, and it is why the numbers are
three orders of magnitude rather than a few percent.

| case | agate | mojo-agate | result |
| --- | ---: | ---: | ---: |
| min | 1884.09 ms | 4.60 ms | 409.67x faster |
| max | 2060.19 ms | 3.77 ms | 545.86x faster |
| sum | 3888.22 ms | 3.36 ms | 1156.61x faster |
| count | 1177.27 ms | 7.46 ms | 157.81x faster |
| mean | 4752.53 ms | 9.85 ms | 482.40x faster |
| variance | 9106.83 ms | 8.82 ms | 1032.31x faster |
| stdev | 4908.37 ms | 6.27 ms | 783.01x faster |
| min (with nulls) | 803.67 ms | 3.70 ms | 217.38x faster |
| median | 7824.71 ms | 194.91 ms | 40.15x faster |
| percentiles(101) | 5610.47 ms | 296.50 ms | 18.92x faster |
| iqr | 5828.98 ms | 208.19 ms | 28.00x faster |
| mad | 14715.44 ms | 211.70 ms | 69.51x faster |

The second table is the fair one: the same float64 work in NumPy, on the same
data, with agate's percentile rule and median rule transliterated so the
comparison is kernel against kernel and not float64 against Decimal.

| case | NumPy float64 | mojo-agate | result |
| --- | ---: | ---: | ---: |
| min | 0.40 ms | 4.44 ms | 0.09x, slower |
| mean | 5.44 ms | 4.62 ms | 1.18x faster |
| variance | 12.09 ms | 6.06 ms | 2.00x faster |
| stdev | 11.65 ms | 5.88 ms | 1.98x faster |
| median | 34.50 ms | 202.40 ms | 0.17x, slower |
| percentiles(101) | 30.06 ms | 194.51 ms | 0.15x, slower |
| mad | 62.41 ms | 319.66 ms | 0.20x, slower |

Read honestly, that says: the two-pass reductions are 2x faster than NumPy,
which is consistent with a serial fused reduction beating a multi-pass SIMD
one; `min` is 11x slower because `np.nanmin` is a heavily optimised SIMD
kernel and this one is a scalar loop with a NaN test; and the order statistics
are 5-6x slower because the sort is the cost, and NumPy uses introsort while
this port uses a bottom-up merge sort. Replacing the merge sort with introsort
would close most of that gap and is the obvious next step; the merge sort is
here because it is stable, which is what makes the ordering agree with
Python's `sorted` on ties.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, compiled to
`dist/libmojo-agate.so` by `build/build.sh`. Buffers cross the C ABI as 64-bit
addresses and are rebuilt in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`.

No kernel allocates: `Pointer.alloc` is gone in Mojo 1.2.0 and a non-raising
`abi("C")` function cannot allocate, so the Python shim owns the sort buffers
and passes their addresses in. `agate_sort` is a bottom-up merge sort that
ping-pongs between two buffers and copies back when the pass count is odd, and
`agate_sort_values` is `values_without_nulls_sorted()` as a single call: drop
the nulls, order the rest, publish the count.

## License

MIT
