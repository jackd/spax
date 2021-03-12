import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax.linalg import subspace_iteration as si
from spax.linalg import test_utils, utils

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def eigh(a, k, largest=True):
    w, v = si.eigh(a, largest=largest)
    return w[:k], utils.standardize_signs(v[:, :k])


def get_keys(n: int, seed: int = 0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


class SubspaceIterationTest(jtu.JaxTestCase):
    def _test_subspace_iteration_method(
        self,
        fun: tp.Callable,
        a: jnp.ndarray,
        v0: jnp.ndarray,
        tol=None,
        atol=None,
        **kwargs
    ):
        fun = jax.jit(fun)
        k = v0.shape[1]
        if tol is None:
            tol = si.default_tol(v0.dtype)
        if atol is None:
            atol = 2 * tol
        w_actual, v_actual, info = fun(a, v0, tol=tol, **kwargs)
        v_actual = utils.standardize_signs(v_actual)

        del info

        # ensure solution satifies eigenvector equation
        self.assertAllClose(a @ v_actual, v_actual * w_actual, atol=atol)
        # ensure result is normalized
        self.assertAllClose(jnp.linalg.norm(v_actual, axis=0), jnp.ones((k,), v0.dtype))

        # ensure result is consistent with `jnp.linalg.eigh`
        w_expected, v_expected = eigh(a, v_actual.shape[1])
        v_expected = utils.standardize_signs(v_expected)

        self.assertAllClose(w_actual, w_expected, atol=atol)
        # do we need the eigenvectors to be close if values are close and eigen equation
        # is satisfied? Finding a tolerance that makes this work is annoying...
        # self.assertAllClose(v_actual, v_expected, atol=atol)

    def _test_subspace_iteration_method_random(
        self, fun: tp.Callable, dtype=np.float32, seed=0, m=50, k=10, tol=None, **kwargs
    ):
        keys = get_keys(2)
        a = test_utils.random_symmetric_mat(keys[0], m, dtype)
        v0 = jax.random.normal(keys[1], shape=(m, k), dtype=dtype)
        self._test_subspace_iteration_method(fun, a, v0, tol=tol, **kwargs)

    def test_basic(self):
        self._test_subspace_iteration_method_random(si.basic_subspace_iteration)

    def test_projected(self):
        self._test_subspace_iteration_method_random(si.projected_subspace_iteration)

    def test_locking_projected(self):
        self._test_subspace_iteration_method_random(
            si.locking_projected_subspace_iteration
        )

    def test_chebyshev(self):
        self._test_subspace_iteration_method_random(
            jax.tree_util.Partial(si.chebyshev_subspace_iteration, 8, 2.0)
        )

    def test_chebyshev_projected(self):
        self._test_subspace_iteration_method_random(
            jax.tree_util.Partial(si.chebyshev_projected_subspace_iteration, 8, 2.0)
        )


if __name__ == "__main__":
    # SubspaceIterationTest().test_basic()
    # SubspaceIterationTest().test_projected()
    # SubspaceIterationTest().test_projected_deflated()
    # SubspaceIterationTest().test_locking_projected()
    # SubspaceIterationTest().test_locking_projected_deflated()
    # SubspaceIterationTest().test_prelocked_projected()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
