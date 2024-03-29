import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized  # pylint: disable=no-name-in-module
from jax import test_util as jtu
from jax.config import config
from jax.experimental.sparse.ops import COO, CSR

from spax import ops
from spax.test_utils import random_uniform

# pylint: disable=undefined-variable

config.parse_flags_with_absl()

ALL_TYPES = (CSR, COO, jnp.ndarray)


class SparseOpsTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                ),
                "shape": shape,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR)
            for shape in ((7, 11), (1, 13), (13, 1))
            for dtype in (np.float32, np.float64)
        )
    )
    def test_to_dense(self, sparse_type, shape, dtype):
        mat = random_uniform(jax.random.PRNGKey(0), shape, dtype=dtype, fmt=jnp.ndarray)
        sp = sparse_type.fromdense(mat)
        redense = ops.to_dense(sp)
        self.assertAllClose(redense, mat)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                ),
                "shape": shape,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for shape in ((7, 11), (1, 13), (13, 1))
            for dtype in (np.float32, np.float64)
        )
    )
    def test_add_sparse(self, sparse_type, shape, dtype):
        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        mat0 = random_uniform(k0, shape, dtype=dtype, fmt=sparse_type)
        mat1 = random_uniform(k1, shape, dtype=dtype, fmt=sparse_type)
        actual = ops.to_dense(ops.add(mat0, mat1))
        expected = ops.to_dense(mat0) + ops.to_dense(mat1)
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}_r{}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                    other_rank,
                ),
                "shape": shape,
                "dtype": dtype,
                "sparse_type": sparse_type,
                "other_rank": other_rank,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for shape in ((7, 11), (1, 13), (13, 1))
            for dtype in (np.float32, np.float64)
            for other_rank in (0, 1, 2, 3)
        )
    )
    def test_add_array(self, sparse_type, shape, other_rank, dtype):
        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        mat = random_uniform(k0, shape, dtype=dtype, fmt=sparse_type)
        if other_rank > len(shape):
            shape = tuple(range(2, 2 + len(shape) - other_rank)) + shape
        else:
            shape = shape[-other_rank:]
        v = jax.random.uniform(k1, shape=shape, dtype=dtype)
        actual = ops.to_dense(ops.add(mat, v))
        expected = ops.to_dense(mat) + v
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}_r{}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                    other_rank,
                ),
                "shape": shape,
                "dtype": dtype,
                "sparse_type": sparse_type,
                "other_rank": other_rank,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for shape in ((7, 11), (1, 13), (13, 1))
            for dtype in (np.float32, np.float64)
            for other_rank in (0, 1, 2, 3)
        )
    )
    def test_mul_array(self, sparse_type, shape, other_rank, dtype):
        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        mat = random_uniform(k0, shape, dtype=dtype, fmt=sparse_type)
        if other_rank > len(shape):
            shape = tuple(range(2, 2 + len(shape) - other_rank)) + shape
        else:
            shape = shape[-other_rank:]
        v = jax.random.uniform(k1, shape=shape, dtype=dtype)
        actual = ops.to_dense(ops.add(mat, v))
        expected = ops.to_dense(mat) + v
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string((nx, ny, nh), dtype),
                    sparse_type.__name__,
                ),
                "nx": nx,
                "ny": ny,
                "nh": nh,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for dtype in (np.float32, np.float64)
            for nx in (5,)
            for ny in (7,)
            for nh in (11,)
        )
    )
    def test_masked_matmul(self, nx, ny, nh, dtype, sparse_type):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        mat = random_uniform(keys[0], (nx, ny), dtype=dtype, fmt=sparse_type)
        x = jax.random.uniform(keys[1], (nx, nh), dtype=dtype)
        y = jax.random.uniform(keys[2], (nh, ny), dtype=dtype)

        actual = ops.to_dense(ops.with_data(mat, ops.masked_matmul(mat, x, y)))
        xt = x @ y
        expected = jnp.where(ops.to_dense(mat) != 0.0, xt, jnp.zeros_like(xt))
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string((nx, ny, nh), dtype),
                    sparse_type.__name__,
                ),
                "nx": nx,
                "ny": ny,
                "nh": nh,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for dtype in (np.float32, np.float64)
            for nx in (5,)
            for ny in (7,)
            for nh in (11,)
        )
    )
    def test_masked_inner(self, nx, ny, nh, dtype, sparse_type):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        mat = random_uniform(keys[0], (nx, ny), dtype=dtype, fmt=sparse_type)
        x = jax.random.uniform(keys[1], (nh, nx), dtype=dtype)
        y = jax.random.uniform(keys[2], (nh, ny), dtype=dtype)

        actual = ops.to_dense(ops.with_data(mat, ops.masked_inner(mat, x, y)))
        xt = x.T @ y
        expected = jnp.where(ops.to_dense(mat) != 0.0, xt, jnp.zeros_like(xt))
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string((nx, ny), dtype),
                    sparse_type.__name__,
                ),
                "nx": nx,
                "ny": ny,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for dtype in (np.float32, np.float64)
            for nx in (5,)
            for ny in (7,)
        )
    )
    def test_masked_outer(self, nx, ny, dtype, sparse_type):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        mat = random_uniform(keys[0], (nx, ny), dtype=dtype, fmt=sparse_type)
        x = jax.random.uniform(keys[1], (nx,), dtype=dtype)
        y = jax.random.uniform(keys[2], (ny,), dtype=dtype)

        actual = ops.to_dense(ops.with_data(mat, ops.masked_outer(mat, x, y)))
        xt = jnp.outer(x, y)
        expected = jnp.where(ops.to_dense(mat) != 0.0, xt, jnp.zeros_like(xt))
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string((size, size), dtype),
                    sparse_type.__name__,
                ),
                "size": size,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for dtype in (np.float32, np.float64)
            for size in (5, 7)
        )
    )
    def test_symmetrize(self, size, dtype, sparse_type):
        mat = random_uniform(
            jax.random.PRNGKey(0), (size, size), dtype=dtype, fmt=sparse_type
        )
        actual = ops.symmetrize(mat)
        expected = ops.to_dense(mat)
        expected = (expected + expected.T) / 2
        self.assertAllClose(ops.to_dense(actual), expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_{sparse_type.__name__}_{axis}",
                "sparse_type": sparse_type,
                "axis": axis,
            }
            for sparse_type in (COO, jnp.ndarray)
            for axis in (0, 1, -1)
        )
    )
    def test_boolean_mask(self, sparse_type, axis):
        shape = (7, 11)
        dtype = jnp.float32
        k0, k1 = jax.random.split(jax.random.PRNGKey(0), 2)
        mat = random_uniform(k0, shape, dtype=dtype, fmt=sparse_type)
        mask = jax.random.uniform(k1, (shape[axis],)) > 0.5
        expected = ops.to_dense(mat)
        if axis == 0:
            expected = expected[mask]
        else:
            expected = expected[:, mask]
        actual = ops.to_dense(ops.boolean_mask(mat, mask, axis=axis))
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_{sparse_type.__name__}_{axis}",
                "sparse_type": sparse_type,
                "axis": axis,
            }
            for sparse_type in (COO, jnp.ndarray)
            for axis in (0, 1, -1)
        )
    )
    def test_gather(self, sparse_type, axis):
        shape = (7, 11)
        dtype = jnp.float32
        k0, k1 = jax.random.split(jax.random.PRNGKey(0), 2)
        mat = random_uniform(k0, shape, dtype=dtype, fmt=sparse_type)
        mask = jax.random.uniform(k1, (shape[axis],)) > 0.5
        (indices,) = jnp.where(mask)
        del mask
        expected = ops.to_dense(mat)
        if axis == 0:
            expected = expected[indices]
        else:
            expected = expected[:, indices]
        actual = ops.to_dense(ops.gather(mat, indices, axis=axis))
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_{sparse_type.__name__}_{axis}",
                "sparse_type": sparse_type,
                "axis": axis,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for axis in (0, 1, -1)
        )
    )
    def test_sum(self, sparse_type, axis):
        shape = (7, 11)
        dtype = jnp.float32
        mat = random_uniform(jax.random.PRNGKey(0), shape, dtype=dtype, fmt=sparse_type)
        expected = ops.sum(mat, axis=axis)
        actual = ops.to_dense(mat).sum(axis=axis)
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_{sparse_type.__name__}_{axis}",
                "sparse_type": sparse_type,
                "axis": axis,
            }
            for sparse_type in (COO, jnp.ndarray)
            for axis in (0, 1, -1)
        )
    )
    def test_max(self, sparse_type, axis):
        shape = (7, 11)
        dtype = jnp.float32
        mat = random_uniform(jax.random.PRNGKey(0), shape, dtype=dtype, fmt=sparse_type)
        expected = ops.max(mat, axis=axis)
        actual = ops.to_dense(mat).max(axis=axis)
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_{sparse_type.__name__}_{axis}_{ord}",
                "sparse_type": sparse_type,
                "axis": axis,
                "ord": ord,
            }
            for sparse_type in (COO, jnp.ndarray)
            for axis in (0, 1, -1)
            for ord in (1, 2, jnp.inf)
        )
    )
    def test_norm(
        self,
        sparse_type,
        ord,
        axis,  # pylint: disable=redefined-builtin
    ):
        shape = (7, 11)
        dtype = jnp.float32
        mat = random_uniform(jax.random.PRNGKey(0), shape, dtype=dtype, fmt=sparse_type)
        mat = ops.map_data(mat, lambda d: d - 0.5)  # make sure we have some negatives
        expected = ops.norm(mat, ord=ord, axis=axis)
        actual = jnp.linalg.norm(ops.to_dense(mat), ord=ord, axis=axis)
        self.assertAllClose(actual, expected)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}".format(
                    jtu.format_shape_dtype_string((nx, ny, nh), dtype),
                    sparse_type.__name__,
                ),
                "nx": nx,
                "ny": ny,
                "nh": nh,
                "dtype": dtype,
                "sparse_type": sparse_type,
            }
            for sparse_type in (COO, CSR, jnp.ndarray)
            for dtype in (np.float32, np.float64)
            for nx in (5,)
            for ny in (7,)
            for nh in (11,)
        )
    )
    def test_masked_outer_rank2(self, nh, nx, ny, dtype, sparse_type):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        mat = random_uniform(keys[0], (nx, ny), dtype=dtype, fmt=sparse_type)
        x = jax.random.uniform(keys[1], (nx, nh), dtype=dtype)
        y = jax.random.uniform(keys[2], (ny, nh), dtype=dtype)

        actual = ops.to_dense(ops.with_data(mat, ops.masked_outer(mat, x, y)))
        xt = x @ y.T
        expected = jnp.where(ops.to_dense(mat) == 0, jnp.zeros_like(xt), xt)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
