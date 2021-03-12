import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax.linalg.solve import minres as mr

config.parse_flags_with_absl()


def simple_sym_ortho(a, b):
    abs_b = jnp.abs(b)
    abs_a = jnp.abs(a)
    if b == 0:
        c = 1.0 if a == 0 else jnp.sign(a)
        s = 0.0
        r = abs_a
    elif a == 0:
        c = 0.0
        s = jnp.sign(b)
        r = abs_b
    elif abs_b >= abs_a:
        tau = a / b
        s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        # abs_a > abs_b
        tau = b / a
        c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


class MinresTest(jtu.JaxTestCase):
    def test_sym_ortho(self):
        # not the most thorough of tests, but simple_sym_ortho is easier to read / check
        def check_consistent(a, b):
            c_actual, s_actual, r_actual = mr.sym_ortho(float(a), float(b))
            c_expected, s_expected, r_expected = simple_sym_ortho(float(a), float(b))
            self.assertAllClose(c_actual, c_expected)
            self.assertAllClose(s_actual, s_expected)
            self.assertAllClose(r_actual, r_expected)

        # branch: b == 0
        check_consistent(0.0, 0.0)
        check_consistent(-0.2, 0.0)
        # branch: a == 0
        check_consistent(0, 1.2)
        check_consistent(0, -1.2)
        # branch: abs_b >= abs_a
        check_consistent(0.5, 0.5)
        check_consistent(0.5, -0.5)
        check_consistent(0.5, -0.7)
        check_consistent(-0.5, -0.7)
        check_consistent(-0.5, 0.7)
        # branch: abs_a > abs_b
        check_consistent(0.6, 0.5)
        check_consistent(-0.6, 0.5)
        check_consistent(-0.6, -0.5)
        check_consistent(0.6, -0.5)


if __name__ == "__main__":
    # MinresTest().test_sym_ortho()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
