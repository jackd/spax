import jax
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax.linalg.lobpcg.basic import lobpcg
from spax.linalg.test_utils import eigh_general, random_symmetric_mat

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


class BasicLobpcgTest(jtu.JaxTestCase):
    def test_lobpcg(self):
        m = 50
        k = 10
        dtype = np.float64
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        a = random_symmetric_mat(keys[0], m, dtype)
        x0 = jax.random.normal(keys[1], (m, k), dtype=dtype)

        B = None

        w_expected, v_expected = eigh_general(a, B, False)
        w_expected = w_expected[:k]
        v_expected = v_expected[:, :k]

        w_actual, v_actual, _ = lobpcg(A=a, B=B, X0=x0, largest=False, max_iters=200)
        self.assertAllClose(w_expected, w_actual, rtol=1e-8, atol=1e-10)
        self.assertAllClose(v_expected, v_actual, rtol=1e-4, atol=1e-10)


if __name__ == "__main__":
    # BasicLobpcgTest().test_eigh_general()
    # BasicLobpcgTest().test_lobpcg()
    # BasicLobpcgTest().test_lobpcg_coo_vjp()
    # BasicLobpcgTest().test_lobpcg_csr_vjp()
    # BasicLobpcgTest().test_lobpcg_simple()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
