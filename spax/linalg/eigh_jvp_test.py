from functools import partial

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax.config import config

from spax import ops, utils
from spax.linalg import eigh_jvp as jvp_lib
from spax.linalg.utils import standardize_signs
from spax.sparse import CSR

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
dtype = jnp.float64


def _eigh_dense(a: jnp.ndarray, w0, v0):
    assert w0.ndim == 1
    assert v0.ndim == 2
    assert w0.shape[0] == v0.shape[1], (w0.shape, v0.shape)
    k = w0.size
    w, v = jnp.linalg.eigh(a)
    w = w[:k]
    v = v[:, :k]
    v = standardize_signs(v)
    return w, v


@jax.custom_jvp
def eigh_dense(a: jnp.ndarray, w0, v0):
    return _eigh_dense(a, w0, v0)


@eigh_dense.defjvp
def _eigh_dense_jvp(primals, tangents):
    a, w0, v0 = primals
    a_dot, w0_dot, v0_dot = tangents
    w, v = eigh_dense(a, w0, v0)

    a = ops.symmetrize(a)
    a_dot = ops.symmetrize(a_dot)

    w_dot, v_dot = jvp_lib.eigh_jvp(a, a_dot, w, v, w0_dot, v0_dot)
    return (w, v), (w_dot, v_dot)


@jax.custom_jvp
def eigh_dense_single(a: jnp.ndarray, w0, v0):
    w, v = _eigh_dense(a, jnp.expand_dims(w0, axis=-1), jnp.expand_dims(v0, axis=-1))
    return jnp.squeeze(w, axis=-1), jnp.squeeze(v, axis=-1)


@eigh_dense_single.defjvp
def _eigh_dense_single_jvp(primals, tangents):
    a, w0, v0 = primals
    a_dot, w_dot0, v_dot0 = tangents
    w, v = eigh_dense_single(a, w0, v0)

    a = ops.symmetrize(a)
    a_dot = ops.symmetrize(a_dot)

    w_dot, v_dot = jvp_lib.eigh_single_jvp(a, a_dot, w, v, w_dot0, v_dot0)
    return (w, v), (w_dot, v_dot)


@jax.custom_jvp
def eigh_csr(indices: jnp.ndarray, indptr: jnp.ndarray, data: jnp.ndarray, w0, v0):
    n = v0.shape[0]
    a = CSR(indices, indptr, data, (n, n))
    return _eigh_dense(ops.to_dense(a), w0, v0)


@eigh_csr.defjvp
def _eigh_csr_jvp(primals, tangents):
    indices, indptr, data, w0, v0 = primals
    _, _, data_dot, w0_dot, v0_dot = tangents
    w, v = eigh_csr(indices, indptr, data, w0, v0)

    n = v0.shape[0]
    a = CSR(indices, indptr, data, (n, n))
    a = ops.symmetrize_data(a)
    a_dot = ops.with_data(a, data_dot)
    a_dot = ops.symmetrize_data(a_dot)

    w_dot, v_dot = jvp_lib.eigh_jvp(a, a_dot, w, v, w0_dot, v0_dot)
    return (w, v), (w_dot, v_dot)


class EighJVPTest(jtu.JaxTestCase):
    def test_eigh_dense_single_jvp(self):
        seed = 0
        n = 16
        jit = True

        a = jax.random.normal(jax.random.PRNGKey(seed), (n, n), dtype=dtype)
        a = a @ a.T  # make SPD
        w0 = jnp.zeros((), dtype=dtype)
        v0 = jnp.zeros((n,), dtype=dtype)

        fun = eigh_dense_single
        if jit:
            fun = jax.jit(fun)
        jtu.check_grads(partial(fun, w0=w0, v0=v0), (a,), order=1)

    def test_eigh_dense_jvp(self):
        seed = 0
        n = 16
        k = 3
        jit = True

        a = jax.random.normal(jax.random.PRNGKey(seed), (n, n), dtype=dtype)
        a = a @ a.T  # make SPD
        w0 = jnp.zeros((k,))
        v0 = jnp.zeros((n, k))

        fun = eigh_dense
        if jit:
            fun = jax.jit(fun)
        jtu.check_grads(partial(fun, w0=w0, v0=v0), (a,), order=1)

    def test_eigh_csr_jvp(self):
        seed = 0
        n = 16
        k = 3
        jit = True

        k0, k1 = jax.random.split(jax.random.PRNGKey(seed), 2)
        a = utils.random_uniform(k0, shape=(n, n), dtype=dtype, fmt="csr")
        diags = jax.random.uniform(k1, shape=(n,), dtype=dtype) + 1
        a = ops.add(a, utils.diag(diags))  # ensure eventual eigenvalues non-degenerate
        a = a.todense()
        a = a @ a.T  # make SPD
        a = CSR.fromdense(a)
        w0 = jnp.zeros((k,))
        v0 = jnp.zeros((n, k))

        fun = eigh_csr
        if jit:
            fun = jax.jit(fun)

        jtu.check_grads(
            partial(fun, a.indices, a.indptr, w0=w0, v0=v0), (a.data,), order=1
        )

        def squared_fun(indices, indptr, data, w0, v0):
            return fun(indices, indptr, data ** 2, w0, v0)

        jtu.check_grads(
            partial(squared_fun, a.indices, a.indptr, w0=w0, v0=v0), (a.data,), order=1
        )


if __name__ == "__main__":
    # absltest.main(testLoader=jtu.JaxTestLoader())
    EighJVPTest().test_eigh_csr_jvp()
    # EighJVPTest().test_eigh_dense_single_jvp()
    # EighJVPTest().test_eigh_dense_jvp()
