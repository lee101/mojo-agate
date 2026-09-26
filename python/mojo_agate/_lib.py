"""ctypes bridge to the compiled Mojo column-statistics kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay `c_int64` for addresses;
`c_int` truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-agate.so"

MIN = 0
MAX = 1
SUM = 2
COUNT = 3
MEAN = 4

VARIANCE = 0
POP_VARIANCE = 1
STDEV = 2
POP_STDEV = 3


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.agate_reduce.restype = None
    lib.agate_reduce.argtypes = [ctypes.c_int64] * 4
    lib.agate_dispersion.restype = None
    lib.agate_dispersion.argtypes = [ctypes.c_int64] * 4
    lib.agate_sort.restype = None
    lib.agate_sort.argtypes = [ctypes.c_int64] * 4
    lib.agate_sort_values.restype = None
    lib.agate_sort_values.argtypes = [ctypes.c_int64] * 6
    lib.agate_median.restype = None
    lib.agate_median.argtypes = [ctypes.c_int64] * 3
    lib.agate_percentiles.restype = None
    lib.agate_percentiles.argtypes = [ctypes.c_int64] * 3
    lib.agate_mad.restype = None
    # (sorted, n, median, res): the median is a Float64
    lib.agate_mad.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_int64
    ]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _values(values):
    return np.ascontiguousarray(values, dtype=np.float64)


def reduce_one(values, code):
    """One scalar reduction: min, max, sum, count or mean."""
    vals = _values(values)
    out = np.zeros(1, dtype=np.float64)
    lib.agate_reduce(_addr(vals), vals.size, _addr(out), code)
    return float(out[0])


def dispersion_one(values, code):
    """Sample or population variance or standard deviation."""
    vals = _values(values)
    out = np.zeros(1, dtype=np.float64)
    lib.agate_dispersion(_addr(vals), vals.size, _addr(out), code)
    return float(out[0])


def sort_values(values):
    """`values_without_nulls_sorted()`: drop NaN, sort ascending.

    Returns the sorted values and how many there are.
    """
    vals = _values(values)
    n = max(vals.size, 1)
    dst = np.zeros(n, dtype=np.float64)
    tmp1 = np.zeros(n, dtype=np.float64)
    tmp2 = np.zeros(n, dtype=np.float64)
    counts = np.zeros(1, dtype=np.float64)
    lib.agate_sort_values(
        _addr(vals), vals.size, _addr(dst), _addr(tmp1), _addr(tmp2),
        _addr(counts),
    )
    count = int(counts[0])
    return dst[:count], count


def median_of_sorted(sorted_values):
    """`agate.utils.median` of an already-sorted array."""
    vals = _values(sorted_values)
    out = np.zeros(1, dtype=np.float64)
    lib.agate_median(_addr(vals), vals.size, _addr(out))
    return float(out[0])


def percentiles_of_sorted(sorted_values):
    """agate's 101 percentiles, 0th through 100th."""
    vals = _values(sorted_values)
    out = np.zeros(101, dtype=np.float64)
    lib.agate_percentiles(_addr(vals), vals.size, _addr(out))
    return out


def mad(sorted_values, median):
    """Median absolute deviation, the way agate computes it.

    `sorted_values` must be the sorted, null-free column, which is what
    upstream iterates over.
    """
    vals = _values(sorted_values)
    out = np.zeros(1, dtype=np.float64)
    lib.agate_mad(_addr(vals), vals.size, ctypes.c_double(median), _addr(out))
    return float(out[0])
