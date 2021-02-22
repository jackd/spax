import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax.linalg import utils
from spax.linalg.utils import as_array_fun

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


z32 = jnp.zeros((), dtype=jnp.float32)


class UtilsTest(jtu.JaxTestCase):
    def test_standardize_signs(self):
        x = jnp.asarray([[1, 2, 3], [-2, 1, 2]])
        expected = jnp.asarray([[-1, 2, 3], [2, 1, 2]])
        actual = utils.standardize_signs(x)
        self.assertAllClose(actual, expected)

    def test_deflate(self):
        n = 10
        dtype = jnp.float32
        a = jax.random.normal(jax.random.PRNGKey(0), shape=(n, n), dtype=dtype)
        a = a @ a.T
        w0, v0 = jnp.linalg.eigh(a)
        deflated = utils.deflate(a, w0[-1], v0[:, -1])
        w1, v1 = jnp.linalg.eigh(deflated)
        v0 = utils.standardize_signs(v0)
        v1 = utils.standardize_signs(v1)
        self.assertAllClose(w1[0], jnp.zeros((), dtype), atol=1e-4)
        self.assertAllClose(w1[1:], w0[:-1], atol=1e-4)
        self.assertAllClose(v1[:, 1:], v0[:, :-1], atol=1e-4)
        self.assertAllClose(v1[:, 0], v0[:, -1], atol=1e-4)

    def test_deflate_fun(self):
        n = 10
        dtype = jnp.float32
        a = jax.random.normal(jax.random.PRNGKey(0), shape=(n, n), dtype=dtype)
        u = jax.random.normal(jax.random.PRNGKey(1), shape=(n,), dtype=dtype)
        s = jnp.asarray(1.2, dtype)
        expected = utils.deflate(a, s, u)
        actual = utils.deflate(as_array_fun(a), s, u)(jnp.eye(n, dtype=dtype))
        self.assertAllClose(actual, expected)

    def test_deflate_eigenvector(self):
        n = 10
        dtype = jnp.float32
        a = jax.random.normal(jax.random.PRNGKey(0), shape=(n, n), dtype=dtype)
        a = a @ a.T
        w0, v0 = jnp.linalg.eigh(a)
        deflated = utils.deflate_eigenvector(a, v0[:, -1])(jnp.eye(n, dtype=dtype))
        w1, v1 = jnp.linalg.eigh(deflated)
        v0 = utils.standardize_signs(v0)
        v1 = utils.standardize_signs(v1)
        self.assertAllClose(w1[0], jnp.zeros((), dtype), atol=1e-4)
        self.assertAllClose(w1[1:], w0[:-1], atol=1e-4)
        self.assertAllClose(v1[:, 1:], v0[:, :-1], atol=1e-4)
        self.assertAllClose(v1[:, 0], v0[:, -1], atol=1e-4)


if __name__ == "__main__":
    # UtilsTest().test_deflate_constant()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
