import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized  # pylint: disable=no-name-in-module
from jax import test_util as jtu
from jax.experimental.sparse_ops import COO, CSR
from jax.test_util import check_grads

from spax.test_utils import random_uniform


# pylint: disable=undefined-variable
class PatchTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_CSR{}_{}".format(
                    ".T" if transpose else "",
                    jtu.format_shape_dtype_string(shape, dtype),
                ),
                "shape": shape,
                "dtype": dtype,
                "transpose": transpose,
            }
            for dtype in (jnp.float32, jnp.float64)
            for shape in ((5,), (5, 7))
            for transpose in (False, True)
        )
    )
    def test_csr_matmul_gradients(self, shape, dtype, transpose):
        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        A_shape = (2, shape[-1])
        if transpose:
            A_shape = A_shape[-1::-1]
        A = random_uniform(k0, A_shape, dtype=dtype, fmt=CSR)
        x = jax.random.normal(k1, shape)

        def f(data, x):
            return CSR((data, A.indices, A.indptr), shape=A.shape) @ x

        check_grads(f, (A.data, x), order=2, atol=1e-3)
        check_grads(lambda x: f(A.data, x), (x,), order=2, atol=1e-3)
        check_grads(lambda d: f(d, x), (A.data,), order=2, atol=1e-3)
        check_grads(lambda z: f(A.data, x), (x,), order=2, atol=1e-3)  # zeros case

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_COO{}_{}".format(
                    ".T" if transpose else "",
                    jtu.format_shape_dtype_string(shape, dtype),
                ),
                "shape": shape,
                "dtype": dtype,
                "transpose": transpose,
            }
            for dtype in (jnp.float32, jnp.float64)
            for shape in ((5,), (5, 7))
            for transpose in (False, True)
        )
    )
    def test_coo_matmul_gradients(self, shape, dtype, transpose):
        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        A_shape = (2, shape[-1])
        if transpose:
            A_shape = A_shape[-1::-1]
        A = random_uniform(k0, A_shape, dtype=dtype, fmt=COO)
        x = jax.random.normal(k1, shape)

        def f(data, x):
            return COO((data, A.row, A.col), shape=A.shape) @ x

        check_grads(f, (A.data, x), order=2, atol=1e-3)
        check_grads(lambda x: f(A.data, x), (x,), order=2, atol=1e-3)
        check_grads(lambda d: f(d, x), (A.data,), order=2, atol=1e-3)
        check_grads(lambda z: f(A.data, x), (x,), order=2, atol=1e-3)  # zeros case


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
