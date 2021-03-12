import jax
import jax.numpy as jnp

from spax import ops, utils
from spax.linalg.solve import cg_least_squares
from spax.linalg.utils import as_array_fun
from spax.sparse import COO, CSR, SparseArray


def project(x, v):
    assert len(x.shape) == 1
    assert len(v.shape) == 1
    return x - v * jnp.dot(v.conj(), x)


def projector(v):
    return jax.tree_util.Partial(project, v=v)


def eigh_rev(grad_w, grad_v, w, v, matmul_fun=jnp.matmul):
    n = w.size
    E = w[jnp.newaxis, :] - w[:, jnp.newaxis]
    vt = v.T

    # grad_a = v (diag(grad_w) + (v^T v.grad / E)) v^T
    #        = v @ inner @ v.T
    inner = jnp.where(jnp.eye(n, dtype=bool), jnp.diag(grad_w), (vt @ grad_v) / E)
    return matmul_fun(v, inner @ vt)


def eigh_single_rev(
    grad_w, grad_v, w, v, x0, a, outer_impl: jnp.outer, tol: float = 1e-5
):
    """
    Args:
        grad_w: [] gradient w.r.t single eigenvalue.
        grad_v: [m] gradient w.r.t single eigenvector.
        w: single eigenvalue.
        v: single eigenvector.
        a: matmul function.
        x0: initial estimate of least squares solution.
        outer_impl: e.g. `coo.masked_outer_fun`.
        tol: tolerance used in least squares.

    Returns:
        grad_data: shape/dtype of `outer_impl` return value ([m, m] array default).
        x0_: solution to least squares problem solved.
    """
    a = as_array_fun(a)
    grad_w * v.conj()

    # def if_zeros(operand):
    #     grad_v, w, v, x0 = operand
    #     del grad_v, v, w
    #     return x0, jnp.zeros_like(g0)

    def otherwise(operand):
        grad_v, w, v, x0 = operand
        # Amat = (a - wi * jnp.eye(m, dtype=a.dtype)).T
        # Projection operator on space orthogonal to v
        proj = projector(v)

        # Find a solution lambda_0 using least-squares conjugate gradient
        # (A - w*I) @ x = proj(grad_v)
        (l0, _) = cg_least_squares(
            lambda x: a(x.conj()).conj() - x * w, proj(grad_v), x0=x0, atol=0, tol=tol
        )
        # Project to correct for round-off errors
        l0 = proj(l0)
        return l0

    operand = grad_v, w, v, x0
    # x0_, g1 = jax.lax.cond(jnp.all(grad_v == 0), if_zeros, otherwise, operand)
    l0 = otherwise(operand)

    z = grad_w * v.conj() - l0
    grad_data = outer_impl(z, v)
    return grad_data, l0


def eigh_partial_rev(
    grad_w, grad_v, w, v, x0, a, outer_impl=jnp.outer, tol: float = 1e-5
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        a: matmul function
        x0: [m, k] initial solution to (A - w[i]I)x[i] = Proj(grad_v[:, i])
        tol: tolerance used in `jnp.linalg.cg`

    Returns:
        grad_a: [m, m] (or output shape/dtyep of `outer_impl`)
        x0: [m, k]
    """
    a = as_array_fun(a)
    grad_a, x0 = jax.vmap(
        jax.tree_util.Partial(eigh_single_rev, a=a, outer_impl=outer_impl, tol=tol),
        in_axes=(0, 1, 0, 1, 1),
        out_axes=(0, 1),
    )(grad_w, grad_v, w, v, x0)
    grad_a = jnp.sum(grad_a, axis=0)
    return grad_a, x0


def _eigh_partial_rev_coo(grad_w, grad_v, w, v, x0, coords, data, tol: float = 1e-5):
    a = COO(coords, data, (v.shape[0],) * 2)
    return eigh_partial_rev(
        grad_w,
        grad_v,
        w,
        v,
        x0,
        a,
        jax.tree_util.Partial(ops.coo.masked_outer, a),
        tol=tol,
    )


def eigh_partial_rev_coo(grad_w, grad_v, w, v, x0, a: COO, tol: float = 1e-5):
    return _eigh_partial_rev_coo(grad_w, grad_v, w, v, x0, a.coords, a.data, tol=tol)


def _eigh_partial_rev_csr(
    grad_w, grad_v, w, v, x0, indices, indptr, data, tol: float = 1e-5
):
    shape = (indptr.shape[0] - 1, v.shape[0])
    a = CSR(indices, indptr, data, shape)
    return eigh_partial_rev(
        grad_w,
        grad_v,
        w,
        v,
        x0,
        a,
        jax.tree_util.Partial(ops.csr.masked_outer, a),
        tol=tol,
    )


def eigh_partial_rev_csr(grad_w, grad_v, w, v, x0, a: CSR, tol: float = 1e-5):
    assert a.shape == (grad_v.shape[0],) * 2
    return _eigh_partial_rev_csr(
        grad_w, grad_v, w, v, x0, a.indices, a.indptr, a.data, tol=tol
    )


def eigh_partial_rev_sparse(
    grad_w, grad_v, w, v, x0, a: SparseArray, tol: float = 1e-5
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        x0: initial solution to least squares problem.
        data, indices, indptr: csr formatted [m, m] matrix.

    Returns:
        grad_data: gradient of `data` input, same `shape` and `dtype`
        x0_: solution to least squares problem.
    """
    if utils.is_coo(a):
        fun = eigh_partial_rev_coo
    elif utils.is_csr(a):
        fun = eigh_partial_rev_csr
    else:
        raise NotImplementedError("Only coo supported")
    return fun(grad_w, grad_v, w, v, x0, a, tol=tol)
