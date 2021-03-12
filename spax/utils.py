import inspect
import typing as tp
from functools import partial, wraps

import jax
import jax.numpy as jnp
from jax._src.ops.scatter import _scatter_update

from spax import sparse


def multiply_leading_dims(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    assert b.ndim >= a.ndim
    assert a.shape == b.shape[: a.ndim]
    return a.reshape(*a.shape, *(1,) * (b.ndim - a.ndim)) * b


def eye(
    N: int, dtype: jnp.dtype = jnp.float32, index_dtype: jnp.dtype = jnp.int32
) -> sparse.COO:
    return diag(jnp.ones((N,), dtype=dtype), index_dtype=index_dtype)


def diag(diagonals: jnp.ndarray, index_dtype: jnp.dtype = jnp.int32) -> sparse.COO:
    """Create a matrix with `diagonals` on the main diagonal."""
    assert diagonals.ndim == 1
    n = diagonals.size
    r = jnp.arange(n, dtype=index_dtype)
    return sparse.COO(jnp.vstack((r, r)), diagonals, (n, n))


_FMTS = {"bsr": sparse.BSR, "coo": sparse.COO, "csr": sparse.CSR, "ell": sparse.ELL}


def _sparse_rng(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        sig = inspect.signature(func).bind(*args, **kwargs)
        sig.apply_defaults()
        arguments = dict(sig.arguments)
        key = arguments.pop("key")
        nnz = arguments["nnz"]
        fmt = arguments["fmt"]

        # TODO(jakevdp): avoid creating dense array.
        key, key2 = jax.random.split(key)
        mat = func(key, **arguments)
        nnz = int(nnz * mat.size if 0 < nnz < 1 else nnz)

        if nnz <= 0:
            mat = jnp.zeros_like(mat)
        elif nnz < mat.size:
            mask = jax.random.shuffle(key2, jnp.arange(mat.size)).reshape(mat.shape)
            mat = jnp.where(mask < nnz, mat, 0)

        if fmt == "dense":
            return mat
        elif fmt in _FMTS:
            return _FMTS[fmt].fromdense(mat)
        else:
            raise ValueError(f"Unrecognized format: {fmt}")

    return wrapped


@_sparse_rng
def random_uniform(
    key: jnp.ndarray,
    shape: tp.Sequence[int] = (),
    dtype: jnp.dtype = jnp.float32,
    minval: tp.Union[float, jnp.ndarray] = 0.0,
    maxval: tp.Union[float, jnp.ndarray] = 1.0,
    nnz: tp.Union[int, float] = 0.1,
    fmt: str = "csr",
) -> sparse.SparseArray:
    """Sparse Uniform Array"""
    return jax.random.uniform(
        key, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


@_sparse_rng
def random_normal(
    key: jnp.ndarray,
    shape: tp.Sequence[int] = (),
    dtype: jnp.dtype = jnp.float32,
    nnz: tp.Union[int, float] = 0.1,
    fmt: str = "csr",
) -> sparse.SparseArray:
    """Sparse Normal Array"""
    return jax.random.normal(key, shape=shape, dtype=dtype)


def canonicalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    assert axis < ndim, (axis, ndim)
    return axis


def segment_max(
    data, segment_ids, num_segments=None, indices_are_sorted=False, initial=None
):
    """Computes the sum within segments of an array.

    Similar to TensorFlow's segment_sum:
    https://www.tensorflow.org/api_docs/python/tf/math/segment_max

    Args:
        data: an array with the values to be summed.
        segment_ids: an array with integer dtype that indicates the segments of
            `data` (along its leading axis) to be summed. Values can be repeated and
            need not be sorted. Values outside of the range [0, num_segments) are
            dropped and do not contribute to the sum.
        num_segments: optional, an int with nonnegative value indicating the number
            of segments. The default is set to be the minimum number of segments that
            would support all indices in ``segment_ids``, calculated as
            ``max(segment_ids) + 1``.
            Since `num_segments` determines the size of the output, a static value
            must be provided to use ``segment_sum`` in a ``jit``-compiled function.
        indices_are_sorted: whether ``segment_ids`` is known to be sorted.
        initial: initial value. If None, ``jnp.iinfo(data.dtype).min`` (or ``finfo``) is
            used.

    Returns:
        An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
        segment sums.
    """

    if num_segments is None:
        num_segments = jnp.max(segment_ids) + 1
    if initial is None:
        if jnp.issubdtype(data.dtype, jnp.integer):
            initial = jnp.iinfo(data.dtype).min
        elif jnp.issubdtype(data.dtype, jnp.floating):
            initial = jnp.finfo(data.dtype).min
        else:
            raise ValueError(f"Unsupported dtype {data.dtype}")
    num_segments = int(num_segments)

    out = jnp.full((num_segments,) + data.shape[1:], initial, dtype=data.dtype)

    return _scatter_update(
        out,
        segment_ids,
        data,
        jax.lax.scatter_max,
        indices_are_sorted,
        unique_indices=False,
        normalize_indices=False,
    )


def segment_softmax(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: tp.Optional[int] = None,
    indices_are_sorted: bool = False,
) -> jnp.ndarray:
    if num_segments is None:
        num_segments = jnp.max(segment_ids) + 1
    max_val = segment_max(data, segment_ids, num_segments, indices_are_sorted)
    data = jnp.exp(data - max_val[segment_ids])
    summed = jax.ops.segment_sum(data, segment_ids, num_segments, indices_are_sorted)
    return data / summed[segment_ids]


def is_trace(x):
    return isinstance(x, jax.core.Trace)


def is_instance_or_abstract(x, concrete_cls, abstract_cls):
    return isinstance(x, concrete_cls) or isinstance(jax.core.get_aval(x), abstract_cls)


def _is_instance_checker(concrete_cls, abstract_cls):
    return partial(
        is_instance_or_abstract, concrete_cls=concrete_cls, abstract_cls=abstract_cls
    )


is_coo = _is_instance_checker(sparse.COO, sparse.AbstractCOO)
is_csr = _is_instance_checker(sparse.CSR, sparse.AbstractCSR)
is_ell = _is_instance_checker(sparse.ELL, sparse.AbstractELL)
is_bsr = _is_instance_checker(sparse.BSR, sparse.AbstractBSR)

is_sparse = _is_instance_checker(
    sparse.SparseArray,
    (sparse.AbstractCOO, sparse.AbstractCSR, sparse.AbstractELL, sparse.AbstractBSR),
)

_is_dense = _is_instance_checker(jnp.ndarray, jax.ShapedArray)


def is_dense(x):
    return not is_sparse(x) and _is_dense(x)
