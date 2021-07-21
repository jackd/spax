from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest  # pylint: disable=no-name-in-module
from jax import test_util as jtu
from jax.config import config
from jax.experimental.sparse.ops import COO, CSR

from spax import ops
from spax.linalg import eigh_vjp as vjp_lib
from spax.linalg.utils import standardize_signs
from spax.test_utils import random_uniform

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def symmetrize(x):
    return (x + x.T) / 2


def eigh_partial(a, k: int, largest: bool):
    w, v = jax.numpy.linalg.eigh(a)
    if largest:
        w = w[-1::-1]
        v = v[:, -1::-1]
    w = w[:k]
    v = v[:, :k]
    v = standardize_signs(v)
    return w, v


class EighJVPTest(jtu.JaxTestCase):
    def test_eigh_vjp(self):
        n = 20
        dtype = np.float64
        a = jax.random.uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype)
        a = symmetrize(a)

        def eigh(a):
            w, v = jax.numpy.linalg.eigh(a)
            v = standardize_signs(v)
            return w, v

        def eigh_fwd(a):
            w, v = eigh(a)
            return (w, v), (w, v)

        def eigh_rev(res, g):
            grad_w, grad_v = g
            w, v = res
            grad_a = vjp_lib.eigh_rev(grad_w, grad_v, w, v)
            grad_a = symmetrize(grad_a)
            return (grad_a,)

        eigh_fun = jax.custom_vjp(eigh)
        eigh_fun.defvjp(eigh_fwd, eigh_rev)
        jtu.check_grads(eigh_fun, (a,), order=1, modes="rev", rtol=1e-3)
        w, v = eigh(a)
        self.assertAllClose(a @ v, v * w, rtol=1e-6)

    def test_eigh_coo_vjp(self):
        n = 20
        dtype = np.float64
        a = random_uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype, fmt="coo")
        a = ops.symmetrize(a)

        def eigh_coo(data, coords, shape):
            a = COO(coords, data, shape)
            a = a.todense()
            w, v = jnp.linalg.eigh(a)
            v = standardize_signs(v)
            return w, v

        def eigh_coo_fwd(data, coords, shape):
            w, v = eigh_coo(data, coords, shape)
            return (w, v), (w, v, data, coords)

        def eigh_coo_rev(res, g):
            grad_w, grad_v = g
            n = grad_v.shape[0]
            shape = (n, n)
            w, v, data, coords = res
            a = COO(coords, data, shape)
            grad_data = vjp_lib.eigh_rev(
                grad_w, grad_v, w, v, jax.tree_util.Partial(ops.masked_matmul, a)
            )
            grad_data = COO(coords, grad_data, shape)
            grad_data = ops.symmetrize_data(grad_data).data
            return (grad_data, None, None)

        eigh = jax.custom_vjp(eigh_coo)
        eigh.defvjp(eigh_coo_fwd, eigh_coo_rev)

        jtu.check_grads(
            partial(eigh, coords=a.coords, shape=a.shape),
            (a.data,),
            order=1,
            modes="rev",
            rtol=1e-3,
        )

    def test_eigh_csr_vjp(self):
        n = 20
        dtype = np.float64
        a = random_uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype, fmt="csr")
        a = ops.symmetrize(a)

        def eigh_csr(data, indices, indptr):
            size = indptr.size - 1
            a = CSR(indices, indptr, data, shape=(size, size)).todense()
            w, v = jnp.linalg.eigh(a)
            v = standardize_signs(v)
            return w, v

        def eigh_csr_fwd(data, indices, indptr):
            w, v = eigh_csr(data, indices, indptr)
            return (w, v), (w, v, indices, indptr, data)

        def eigh_csr_rev(res, g):
            grad_w, grad_v = g
            w, v, indices, indptr, data = res
            size = indptr.size - 1
            a = CSR(indices, indptr, data, shape=(size, size))
            grad_data = vjp_lib.eigh_rev(
                grad_w, grad_v, w, v, jax.tree_util.Partial(ops.masked_matmul, a)
            )
            grad_data = ops.with_data(a, grad_data)
            grad_data = ops.symmetrize_data(grad_data).data
            return (grad_data, None, None)

        eigh = jax.custom_vjp(eigh_csr)
        eigh.defvjp(eigh_csr_fwd, eigh_csr_rev)

        jtu.check_grads(
            partial(eigh, indices=a.indices, indptr=a.indptr),
            (a.data,),
            order=1,
            modes="rev",
            rtol=1e-3,
        )

    def test_eigh_partial_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = jax.random.uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype)
        a = symmetrize(a)

        def eigh_partial_fwd(a, k: int, largest: bool):
            w, v = eigh_partial(a, k, largest)
            return (w, v), (w, v, a)

        def eigh_partial_rev(res, g):
            w, v, a = res
            grad_w, grad_v = g
            rng_key = jax.random.PRNGKey(0)
            x0 = jax.random.normal(rng_key, v.shape, dtype=v.dtype)
            grad_a, x0 = vjp_lib.eigh_partial_rev(grad_w, grad_v, w, v, x0, a)
            grad_a = symmetrize(grad_a)
            return (grad_a, None, None)

        eigh_partial_fun = jax.custom_vjp(eigh_partial)
        eigh_partial_fun.defvjp(eigh_partial_fwd, eigh_partial_rev)

        jtu.check_grads(
            partial(eigh_partial_fun, k=k, largest=largest),
            (a,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )

        def squared_fun(a):
            a = jnp.exp(a)
            return eigh_partial_fun(a, k=k, largest=largest)

        jtu.check_grads(squared_fun, (a,), 1, modes=["rev"])

    def test_eigh_partial_coo_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = random_uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype, fmt="coo")
        a = ops.symmetrize(a)

        def eigh_partial_coo(data, coords, size, k: int, largest: bool):
            a = COO(coords, data, (size, size)).todense()
            w, v = eigh_partial(a, k, largest)
            v = standardize_signs(v)
            return w, v

        def eigh_partial_fwd(data, coords, size, k: int, largest: bool):
            w, v = eigh_partial_coo(data, coords, size, k, largest)
            return (w, v), (w, v, data, coords)

        def eigh_partial_rev(res, g):
            w, v, data, coords = res
            size = v.shape[0]
            a = COO(coords, data, (size, size))
            grad_w, grad_v = g
            x0 = jax.random.normal(jax.random.PRNGKey(0), v.shape, dtype=dtype)
            grad_data, _ = vjp_lib.eigh_partial_rev(
                grad_w, grad_v, w, v, x0, a, jax.tree_util.Partial(ops.masked_outer, a)
            )

            grad_data = ops.with_data(a, grad_data)
            grad_data = ops.symmetrize_data(grad_data).data
            return grad_data, None, None, None, None

        eigh_partial_fn = jax.custom_vjp(eigh_partial_coo)
        eigh_partial_fn.defvjp(eigh_partial_fwd, eigh_partial_rev)

        jtu.check_grads(
            partial(eigh_partial_fn, k=k, largest=largest, coords=a.coords, size=n),
            (a.data,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )

    def test_eigh_partial_csr_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = random_uniform(jax.random.PRNGKey(0), (n, n), dtype=dtype, fmt="csr")
        a = ops.symmetrize(a)

        def eigh_partial_coo(data, indices, indptr, k: int, largest: bool):
            size = indptr.size - 1
            a = ops.symmetrize_data(CSR(indices, indptr, data, (size, size))).todense()
            w, v = eigh_partial(a, k, largest)
            v = standardize_signs(v)
            return w, v

        def eigh_partial_fwd(data, indices, indptr, k: int, largest: bool):
            w, v = eigh_partial_coo(data, indices, indptr, k, largest)
            return (w, v), (w, v, data, indices, indptr)

        def eigh_partial_rev(res, g):
            w, v, data, indices, indptr = res
            grad_w, grad_v = g
            size = indptr.size - 1
            a = CSR(indices, indptr, data, shape=(size, size))
            x0 = jax.random.normal(jax.random.PRNGKey(0), v.shape, dtype=dtype)
            grad_data, _ = vjp_lib.eigh_partial_rev(
                grad_w, grad_v, w, v, x0, a, jax.tree_util.Partial(ops.masked_outer, a)
            )
            grad_data = ops.symmetrize_data(ops.with_data(a, grad_data)).data
            return grad_data, None, None, None, None

        eigh_partial_fn = jax.custom_vjp(eigh_partial_coo)
        eigh_partial_fn.defvjp(eigh_partial_fwd, eigh_partial_rev)

        jtu.check_grads(
            partial(
                eigh_partial_fn,
                k=k,
                largest=largest,
                indices=a.indices,
                indptr=a.indptr,
            ),
            (a.data,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )


if __name__ == "__main__":
    # EighJVPTest().test_eigh_vjp()
    # EighJVPTest().test_eigh_coo_vjp()
    # EighJVPTest().test_eigh_csr_vjp()
    # EighJVPTest().test_eigh_partial_vjp()
    # EighJVPTest().test_eigh_partial_coo_vjp()
    # EighJVPTest().test_eigh_partial_csr_vjp()
    absltest.main(testLoader=jtu.JaxTestLoader())
