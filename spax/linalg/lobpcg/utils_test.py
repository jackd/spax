import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax.linalg.lobpcg import utils

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def symmetrize(A):
    return (A + A.T) / 2


class LobpcgUtilsTest(jtu.JaxTestCase):
    def assertAllLessEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        for ai, bi in zip(a, b):
            self.assertLessEqual(ai, bi)

    def assertAllGreaterEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        for ai, bi in zip(a, b):
            self.assertGreaterEqual(ai, bi)

    # def test_approx_matrix_norm2(self):
    #     m = 600
    #     nx = 10
    #     rng = np.random.default_rng(1)
    #     A = symmetrize(rng.normal(size=(m, m)))
    #     x = rng.normal(size=(m, nx))

    #     actual = utils.approx_matrix_norm2(A, x)
    #     expected = jnp.linalg.norm(A, 2)
    #     self.assertAllClose(actual, expected)

    def test_eigh(self):
        m = 10
        rng = np.random.default_rng(0)
        A = symmetrize(rng.normal(size=(m, m)))
        w, v = utils.eigh(A, largest=True)
        self.assertAllClose(A @ v, v * w, rtol=1e-10)
        self.assertAllGreaterEqual(w[:-1], w[1:])
        w, v = utils.eigh(A, largest=False)
        self.assertAllClose(A @ v, v * w, rtol=1e-10)
        self.assertAllLessEqual(w[:-1], w[1:])

    def test_rayleigh_ritz(self):
        m = 20
        nx = 5

        rng = np.random.default_rng(0)
        A = symmetrize(rng.normal(size=(m, m)))
        B = symmetrize(rng.normal(size=(m, m)))
        B += m * np.eye(m)
        S = rng.normal(size=(m, nx))

        E, C = utils.rayleigh_ritz(S, A, B, largest=True)
        self.assertAllClose(C.T @ S.T @ B @ S @ C, jnp.eye(nx), atol=1e-12)
        self.assertAllClose(C.T @ S.T @ A @ S @ C, jnp.diag(E), atol=1e-12)
        self.assertAllGreaterEqual(E[:-1], E[1:])

        E, C = utils.rayleigh_ritz(S, A, B, largest=False)
        self.assertAllClose(C.T @ S.T @ B @ S @ C, jnp.eye(nx), atol=1e-12)
        self.assertAllClose(C.T @ S.T @ A @ S @ C, jnp.diag(E), atol=1e-12)
        self.assertAllLessEqual(E[:-1], E[1:])

    def test_rayleigh_ritz_modified_ortho(self):
        m = 20
        nx = 5
        dtype = np.float32
        largest = False
        tols = dict(rtol=1e-5, atol=1e-5)

        tau_ortho = utils.default_tol(dtype)

        rng = np.random.default_rng(0)
        A = symmetrize(rng.normal(size=(m, m))).astype(dtype)
        X = rng.normal(size=(m, nx)).astype(dtype)
        B = lambda x: x
        nc = 1

        kwargs = dict(A=A, nx=nx, nc=nc)

        E, C = utils.rayleigh_ritz(X, A, B=None, largest=largest)
        X = X @ C
        R = A @ X - X * E
        W = utils.ortho_drop(B, R, X, tau_ortho, tau_ortho, tau_ortho)

        S = jnp.concatenate((X, W), axis=1)
        BS = B(S)
        self.assertAllClose(S.T @ BS, jnp.eye(S.shape[1], dtype=dtype), **tols)
        theta_x, theta_p, cx, cp = utils.rayleigh_ritz_modified_ortho(
            S, largest=largest, **kwargs
        )
        AS = A @ S
        self.assertAllClose(cx.T @ S.T @ AS @ cx, jnp.diag(theta_x), **tols)
        self.assertAllClose(cp.T @ S.T @ AS @ cp, theta_p, **tols)
        self.assertAllClose(cx.T @ S.T @ BS @ cx, jnp.eye(nx, dtype=dtype), **tols)
        self.assertAllClose(cp.T @ S.T @ BS @ cp, jnp.eye(nx - nc, dtype=dtype), **tols)


if __name__ == "__main__":
    # LobpcgUtilsTest().test_rayleigh_ritz_modified_ortho()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
