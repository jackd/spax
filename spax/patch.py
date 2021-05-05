import jax
from jax.experimental import sparse_ops


def _grad_components(aval: jax.ShapedArray, *pairs):
    def grad_component(x, fn):
        return None if isinstance(x, jax.ad_util.Zero) else fn()

    components = (grad_component(*pair) for pair in pairs)
    components = [c for c in components if c is not None]
    if components:
        return sum(components)
    return jax.ad.Zero(aval)


_csr_matmat = sparse_ops.csr_matmat_p.impl


def csr_matmat_jvp(primals, tangents, *, shape, transpose=False):
    data, indices, indptr, B = primals
    d_data, _, _, d_B = tangents
    kwargs = dict(shape=shape, transpose=transpose)
    result = _csr_matmat(data, indices, indptr, B, **kwargs)
    grad = _grad_components(
        jax.ShapedArray((shape[1] if transpose else shape[0], B.shape[1]), data.dtype),
        (d_B, lambda: _csr_matmat(data, indices, indptr, d_B, **kwargs)),
        (d_data, lambda: _csr_matmat(d_data, indices, indptr, B, **kwargs)),
    )
    return result, grad


jax.ad.primitive_jvps[sparse_ops.csr_matmat_p] = csr_matmat_jvp

_csr_matvec = sparse_ops.csr_matvec_p.impl


def csr_matvec_jvp(primals, tangents, *, shape, transpose=False):
    data, indices, indptr, B = primals
    d_data, _, _, d_B = tangents
    kwargs = dict(shape=shape, transpose=transpose)

    result = _csr_matvec(data, indices, indptr, B, **kwargs)
    grad = _grad_components(
        jax.ShapedArray((shape[1] if transpose else shape[0],), data.dtype),
        (d_B, lambda: _csr_matvec(data, indices, indptr, d_B, **kwargs)),
        (d_data, lambda: _csr_matvec(d_data, indices, indptr, B, **kwargs)),
    )
    return result, grad


jax.ad.primitive_jvps[sparse_ops.csr_matvec_p] = csr_matvec_jvp


_coo_matmat = sparse_ops.coo_matmat_p.impl


def coo_matmat_jvp(primals, tangents, *, shape, transpose=False):
    data, row, col, B = primals
    d_data, _, _, d_B = tangents
    kwargs = dict(shape=shape, transpose=transpose)
    result = _coo_matmat(data, row, col, B, **kwargs)
    grad = _grad_components(
        jax.ShapedArray((shape[1] if transpose else shape[0], B.shape[1]), data.dtype),
        (d_B, lambda: _coo_matmat(data, row, col, d_B, **kwargs)),
        (d_data, lambda: _coo_matmat(d_data, row, col, B, **kwargs)),
    )
    return result, grad


jax.ad.primitive_jvps[sparse_ops.coo_matmat_p] = coo_matmat_jvp

_coo_matvec = sparse_ops.coo_matvec_p.impl


def coo_matvec_jvp(primals, tangents, *, shape, transpose=False):
    data, row, col, B = primals
    d_data, _, _, d_B = tangents
    kwargs = dict(shape=shape, transpose=transpose)

    result = _coo_matvec(data, row, col, B, **kwargs)
    grad = _grad_components(
        jax.ShapedArray((shape[1] if transpose else shape[0],), data.dtype),
        (d_B, lambda: _coo_matvec(data, row, col, d_B, **kwargs)),
        (d_data, lambda: _coo_matvec(d_data, row, col, B, **kwargs)),
    )
    return result, grad


jax.ad.primitive_jvps[sparse_ops.coo_matvec_p] = coo_matvec_jvp
