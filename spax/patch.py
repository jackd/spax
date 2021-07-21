import jax
from jax.experimental.sparse import ops
from jax.interpreters.partial_eval import Zero


def _grad_components(aval: jax.ShapedArray, *pairs):
    def grad_component(x, fn):
        return None if isinstance(x, Zero) else fn()

    components = (grad_component(*pair) for pair in pairs)
    components = [c for c in components if c is not None]
    if components:
        return sum(components)
    return Zero(aval)


_csr_matmat = ops.csr_matmat_p.impl


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


jax.ad.primitive_jvps[ops.csr_matmat_p] = csr_matmat_jvp

_csr_matvec = ops.csr_matvec_p.impl


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


jax.ad.primitive_jvps[ops.csr_matvec_p] = csr_matvec_jvp


_coo_matmat = ops.coo_matmat_p.impl


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


jax.ad.primitive_jvps[ops.coo_matmat_p] = coo_matmat_jvp

_coo_matvec = ops.coo_matvec_p.impl


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


jax.ad.primitive_jvps[ops.coo_matvec_p] = coo_matvec_jvp


def _asarray(x):
    return (
        x
        if isinstance(x, (jax.core.ShapedArray, jax.api.ShapeDtypeStruct))
        else jax.numpy.asarray(x)
    )


def _coo_init(self, args, *, shape):
    self.data, self.row, self.col = map(_asarray, args,)
    ops.JAXSparse.__init__(self, args, shape=shape)


ops.COO.__init__ = _coo_init


def _csx_init(self, args, *, shape):
    self.data, self.indices, self.indptr = map(_asarray, args)
    ops.JAXSparse.__init__(self, args, shape=shape)


ops.CSR.__init__ = _csx_init
ops.CSC.__init__ = _csx_init
