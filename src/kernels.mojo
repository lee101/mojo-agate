"""Column statistics: the numeric core of agate's aggregations.

agate is a table library. Nearly all of it is IO (CSV/JSON readers and
writers), table plumbing (pivot, join, group_by, order_by) and formatting
(`babel` number and date rendering). The part that does arithmetic is
`agate/aggregations/`, and it is all reductions over one Number column:

- `Min`, `Max`, `Sum`, `Count`, `Mean`        single-pass reductions
- `Variance`, `PopulationVariance`            two-pass, /(n-1) and /n
- `StDev`, `PopulationStDev`                  sqrt of the two above
- `Median` -> `Percentiles[50]`               order statistics
- `Percentiles`                               agate's own CDF method
- `IQR`, `MAD`, `Quartiles`, `Quintiles`,
  `Deciles`                                   derived from percentiles
- `utils.median`                              the even/odd midpoint rule

Those are the kernels here. agate evaluates them in `decimal.Decimal`, so the
results agree with these float64 kernels to rounding, not bit-for-bit; the
parity tests use exactly representable data and a tight relative tolerance.

Nulls are NaN in the buffer, matching the way agate drops nulls before
reducing: every kernel skips them, exactly as `values_without_nulls()` does.

Buffers cross the C ABI as 64-bit addresses, `@export` rejects parametric
functions, and `AnyOrigin[mut=True]` is the only usable mutable origin, so each
export takes `Int` addresses and rebuilds its own pointer.
"""

from std.math import ceil, floor, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]

# reduction codes, shared with the Python shim
comptime R_MIN: Int = 0
comptime R_MAX: Int = 1
comptime R_SUM: Int = 2
comptime R_COUNT: Int = 3
comptime R_MEAN: Int = 4

# dispersion codes, shared with the Python shim
comptime D_VARIANCE: Int = 0
comptime D_POP_VARIANCE: Int = 1
comptime D_STDEV: Int = 2
comptime D_POP_STDEV: Int = 3


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def nan() -> Float64:
    return Float64(0.0) / Float64(0.0)


def merge_runs(a: FPtr, b: FPtr, n: Int, width: Int):
    """One bottom-up merge pass of runs `width` long: `a` into `b`."""
    var i = 0
    while i < n:
        var mid = min(i + width, n)
        var stop = min(i + 2 * width, n)
        var lo = i
        var hi = mid
        var out = i
        while lo < mid and hi < stop:
            # `<=` keeps the sort stable, as Python's sorted() is
            if a[unsafe_offset=lo] <= a[unsafe_offset=hi]:
                b[unsafe_offset=out] = a[unsafe_offset=lo]
                lo += 1
            else:
                b[unsafe_offset=out] = a[unsafe_offset=hi]
                hi += 1
            out += 1
        while lo < mid:
            b[unsafe_offset=out] = a[unsafe_offset=lo]
            lo += 1
            out += 1
        while hi < stop:
            b[unsafe_offset=out] = a[unsafe_offset=hi]
            hi += 1
            out += 1
        i = stop


@export("agate_reduce")
def agate_reduce(x_addr: Int, n: Int, res_addr: Int, code: Int) abi("C"):
    """Min, max, sum, count or mean of the non-NaN values of `x`.

    agate's `Min`/`Max`/`Sum`/`Mean` return None for an all-null column and
    `Count` returns 0; that is the one difference in this function, and it is
    reproduced here rather than papered over with NaN.
    """
    var x = fp(x_addr)
    var res = fp(res_addr)
    var cnt = 0
    var total = Float64(0.0)
    var lo = Float64(0.0)
    var hi = Float64(0.0)
    for i in range(n):
        var v = x[unsafe_offset=i]
        if v == v:  # not NaN
            cnt += 1
            total += v
            if cnt == 1:
                lo = v
                hi = v
            else:
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
    var r = nan()
    if code == R_COUNT:
        r = Float64(cnt)
    elif cnt > 0:
        if code == R_MIN:
            r = lo
        elif code == R_MAX:
            r = hi
        elif code == R_SUM:
            r = total
        elif code == R_MEAN:
            r = total / Float64(cnt)
    res[unsafe_offset=0] = r

@export("agate_dispersion")
def agate_dispersion(x_addr: Int, n: Int, res_addr: Int, code: Int) abi("C"):
    """Sample and population variance and standard deviation.

    Two passes, like agate: the mean first, then the sum of squared deviations
    about it, divided by n-1 or n. The sample form needs at least two values,
    so a one-value column is NaN, which is what dividing by zero in
    `Variance.run` amounts to.
    """
    var x = fp(x_addr)
    var res = fp(res_addr)
    var cnt = 0
    var total = Float64(0.0)
    for i in range(n):
        var v = x[unsafe_offset=i]
        if v == v:
            cnt += 1
            total += v
    var r = nan()
    if cnt > 0:
        var mean = total / Float64(cnt)
        var ss = Float64(0.0)
        for i in range(n):
            var v = x[unsafe_offset=i]
            if v == v:
                var d = v - mean
                ss += d * d
        if code == D_POP_VARIANCE or code == D_POP_STDEV:
            r = ss / Float64(cnt)
            if code == D_POP_STDEV:
                r = sqrt(r)
        elif cnt > 1:
            r = ss / Float64(cnt - 1)
            if code == D_STDEV:
                r = sqrt(r)
    res[unsafe_offset=0] = r


@export("agate_sort")
def agate_sort(src_addr: Int, n: Int, dst_addr: Int, tmp_addr: Int) abi("C"):
    """Bottom-up merge sort of `n` values, ascending, result in `dst`.

    Stable, so equal values keep their input order the way Python's `sorted`
    does, and O(n log n) worst case. `tmp` is a second buffer of `n` elements.
    agate sorts before every order statistic (`values_without_nulls_sorted`),
    so this is on the critical path of median, percentiles and MAD.
    """
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var tmp = fp(tmp_addr)
    for i in range(n):
        tmp[unsafe_offset=i] = src[unsafe_offset=i]
    var width = 1
    var to_tmp = False
    while width < n:
        if to_tmp:
            merge_runs(dst, tmp, n, width)
        else:
            merge_runs(tmp, dst, n, width)
        to_tmp = not to_tmp
        width = width * 2
    # an odd number of passes leaves the result in tmp, an even one in dst
    if not to_tmp:
        for i in range(n):
            dst[unsafe_offset=i] = tmp[unsafe_offset=i]


@export("agate_sort_values")
def agate_sort_values(x_addr: Int, n: Int, dst_addr: Int, tmp1_addr: Int,
                      tmp2_addr: Int, count_addr: Int) abi("C"):
    """Compact the non-NaN values of `x` into `dst` and sort them.

    This is agate's `values_without_nulls_sorted()`: drop the nulls, order the
    rest, and publish how many survived. Only the first `count` entries of
    `dst` are meaningful on return.
    """
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    var counts = fp(count_addr)
    var cnt = 0
    for i in range(n):
        var v = x[unsafe_offset=i]
        if v == v:
            dst[unsafe_offset=cnt] = v
            cnt += 1
    counts[unsafe_offset=0] = Float64(cnt)
    # agate_sort leaves the result in its `dst` argument, which here is tmp1;
    # copy it back so the sorted values are where the caller expects them
    agate_sort(dst_addr, cnt, tmp1_addr, tmp2_addr)
    for i in range(cnt):
        dst[unsafe_offset=i] = fp(tmp1_addr)[unsafe_offset=i]


@export("agate_median")
def agate_median(sorted_addr: Int, n: Int, res_addr: Int) abi("C"):
    """`agate.utils.median` of already-sorted values.

    Odd length takes the middle element, even length averages the two middle
    ones. An empty column is None upstream and NaN here.
    """
    var s = fp(sorted_addr)
    var res = fp(res_addr)
    var r = nan()
    if n > 0:
        if n % 2 == 1:
            r = s[unsafe_offset=(n + 1) // 2 - 1]
        else:
            var half = n // 2
            r = (s[unsafe_offset=half - 1] + s[unsafe_offset=half]) / 2.0
    res[unsafe_offset=0] = r


@export("agate_percentiles")
def agate_percentiles(sorted_addr: Int, n: Int, res_addr: Int) abi("C"):
    """Agate's `Percentiles`: 101 values, 0th through 100th, CDF method.

    Reproduced exactly, including the awkward parts: the rank is
    `k = n * (p / 100)`, `low = max(1, ceil(k))`, `high = min(n, floor(k + 1))`,
    and when they differ the answer is the mean of `data[low - 1]` and
    `data[high - 1]`. The 0th percentile is the first datum and the 100th is
    the last, whatever the rank arithmetic would say. An empty column is 101
    nulls upstream, so every result here is NaN.
    """
    var s = fp(sorted_addr)
    var res = fp(res_addr)
    for j in range(101):
        res[unsafe_offset=j] = nan()
    if n <= 0:
        return
    res[unsafe_offset=0] = s[unsafe_offset=0]
    for p in range(1, 100):
        var k = Float64(n) * (Float64(p) / 100.0)
        var low = max(1, Int(ceil(k)))
        var high = min(n, Int(floor(k + 1.0)))
        var value = nan()
        if low == high:
            value = s[unsafe_offset=low - 1]
        else:
            value = (s[unsafe_offset=low - 1] + s[unsafe_offset=high - 1]) / 2.0
        res[unsafe_offset=p] = value
    res[unsafe_offset=100] = s[unsafe_offset=n - 1]


@export("agate_mad")
def agate_mad(sorted_addr: Int, n: Int, median: Float64,
              res_addr: Int) abi("C"):
    """`agate.aggregations.MAD`, reproduced exactly.

    Upstream computes `median(tuple(abs(n - m) for n in data))` where `data`
    is the *sorted* column and `median` is `agate.utils.median`, which
    requires sorted input. The deviations are not sorted, so for an even
    length upstream averages the third and fourth deviation *in column
    order*, not the two middle ones. That is not the textbook median absolute
    deviation; it is what agate returns, and parity is the point of this
    port, so the same rule is applied here. The input is the sorted column,
    as upstream's is.
    """
    var s = fp(sorted_addr)
    var res = fp(res_addr)
    if n <= 0:
        res[unsafe_offset=0] = nan()
        return
    var half = n // 2
    var r = Float64(0.0)
    if n % 2 == 1:
        var d = s[unsafe_offset=(n + 1) // 2 - 1] - median
        if d < 0.0:
            d = -d
        r = d
    else:
        var a = s[unsafe_offset=half - 1] - median
        var b = s[unsafe_offset=half] - median
        if a < 0.0:
            a = -a
        if b < 0.0:
            b = -b
        r = (a + b) / 2.0
    res[unsafe_offset=0] = r
