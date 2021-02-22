# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=undefined-variable,fixme,unnecessary-lambda


import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import dtypes
from jax import test_util as jtu
from jax import xla
from jax.config import config, flags

from spax import sparse, utils

config.parse_flags_with_absl()

FLAGS = flags.FLAGS


def rng_sparse(rng, shape, dtype, nnz=0.2):
    mat = rng(shape, dtype)
    size = int(np.prod(shape))
    if 0 < nnz < 1:
        nnz = nnz * size
    nnz = int(nnz)
    if nnz == 0:
        return np.zeros_like(mat)
    elif nnz >= size:
        return mat
    else:
        # TODO(jakevdp): do we care about duplicates?
        cutoff = np.sort(mat.ravel())[nnz]
        mat[mat >= cutoff] = 0
        return mat


test_shapes_2d = [(1, 5), (2, 3), (10, 1)]
test_shapes = {
    sparse.BSR: test_shapes_2d,
    sparse.CSR: test_shapes_2d,
    sparse.ELL: test_shapes_2d,
    sparse.COO: [(5,), (2, 3), (4, 2, 5)],
}


class SparseLaxTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}".format(sparse_type.__name__),
                "sparse_type": sparse_type,
            }
            for sparse_type in [sparse.BSR, sparse.COO, sparse.CSR, sparse.ELL]
        )
    )
    def testAbstractify(self, sparse_type):
        x_dense = jnp.arange(100).reshape(10, 10) % 2
        x_sparse = sparse_type.fromdense(x_dense)
        x_aval = xla.abstractify(x_sparse)
        self.assertEqual(x_aval.shape, x_sparse.shape)
        self.assertEqual(x_aval.dtype, x_sparse.dtype)
        self.assertEqual(x_aval.nnz, x_sparse.nnz)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}".format(sparse_type.__name__),
                "sparse_type": sparse_type,
            }
            for sparse_type in [sparse.BSR, sparse.COO, sparse.CSR, sparse.ELL]
        )
    )
    def testBufferAccess(self, sparse_type):
        x_dense = jnp.arange(100).reshape(10, 10) % 2
        x_sparse = sparse_type.fromdense(x_dense)
        args_maker = lambda: [x_sparse]
        get_data = lambda x: x.data
        self._CompileAndCheck(get_data, args_maker)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}_nnz={}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                    nnz,
                ),
                "dtype": dtype,
                "shape": shape,
                "sparse_type": sparse_type,
                "nnz": nnz,
            }
            for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]
            for sparse_type in [sparse.COO, sparse.CSR, sparse.ELL, sparse.BSR]
            for nnz in [0, 0.2, 0.8]
            for shape in test_shapes[sparse_type]
        )
    )
    def testDenseRoundTrip(self, dtype, shape, sparse_type, nnz):
        rng = jtu.rand_default(self.rng())
        dtype = dtypes.canonicalize_dtype(dtype)
        x_dense = rng_sparse(rng, shape, dtype, nnz=nnz)
        x_sparse = sparse_type.fromdense(x_dense)

        self.assertEqual(x_sparse.nnz, (x_dense != 0).sum())
        self.assertArraysEqual(x_sparse.todense(), x_dense)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}_nnz={}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    sparse_type.__name__,
                    nnz,
                ),
                "dtype": dtype,
                "shape": shape,
                "sparse_type": sparse_type,
                "nnz": nnz,
            }
            for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]
            for sparse_type in [sparse.COO, sparse.CSR, sparse.ELL, sparse.BSR]
            for nnz in [0, 0.2, 0.8]
            for shape in test_shapes_2d
        )
    )
    def testMatvec(self, dtype, shape, sparse_type, nnz):
        rng = jtu.rand_default(self.rng())
        dtype = dtypes.canonicalize_dtype(dtype)
        args_maker = lambda: [
            rng_sparse(rng, shape, dtype, nnz),
            rng(shape[-1:], dtype),
        ]

        np_fun = lambda a, b: np.dot(a, b)
        jnp_fun = lambda a, b: sparse_type.fromdense(a).matvec(b)
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol={np.float16: 2e-3})

        # Check sparse compilation.
        # TODO(jakevdp): enable this test after defining translation rules.
        # args = args_maker()
        # args[0] = sparse_type.fromdense(args[0])
        # jnp_fun = lambda a, b: a.matvec(b)
        # result1 = jnp_fun(*args)
        # result2 = jit(jnp_fun)(*args)
        # self.assertEqual(result1.shape, result2.shape)
        # self.assertEqual(result1.dtype, result2.dtype)
        # self.assertAllClose(result1.data, result2.data)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_{}->{}_nnz={}".format(
                    jtu.format_shape_dtype_string(shape, dtype),
                    input_type.__name__,
                    output_type.__name__,
                    nnz,
                ),
                "dtype": dtype,
                "shape": shape,
                "input_type": input_type,
                "output_type": output_type,
                "nnz": nnz,
            }
            for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]
            for input_type in [sparse.COO, sparse.CSR, sparse.ELL, sparse.BSR]
            for output_type in [sparse.COO, sparse.CSR, sparse.ELL, sparse.BSR]
            for nnz in [0, 0.2, 0.8]
            for shape in test_shapes_2d
        )
    )
    def testFormatConversions(self, dtype, shape, input_type, output_type, nnz):
        rng = jtu.rand_default(self.rng())
        dtype = dtypes.canonicalize_dtype(dtype)
        x_dense = rng_sparse(rng, shape, dtype, nnz=nnz)
        x_input = input_type.fromdense(x_dense)
        expected = output_type.fromdense(x_dense)
        actual = getattr(x_input, f"to{output_type.__name__.lower()}")()

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.index_dtype, actual.index_dtype)
        self.assertEqual(expected.nnz, actual.nnz)
        self.assertArraysEqual(expected.todense(), actual.todense())


class SparseRandomTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": "_{}_fmt={}_nnz={}".format(
                    jtu.format_shape_dtype_string(shape, dtype), fmt, nnz
                ),
                "dtype": dtype,
                "shape": shape,
                "nnz": nnz,
                "fmt": fmt,
            }
            for dtype in [np.float16, np.float32]
            for shape in test_shapes_2d
            for nnz in [0, 0.1, 20]
            for fmt in ["bsr", "coo", "csr", "ell"]
        )
    )
    def testUniform(self, dtype, shape, nnz, fmt):
        key = jax.random.PRNGKey(1701)
        arr = utils.random_uniform(key, shape=shape, dtype=dtype, fmt="dense", nnz=nnz)
        arr_sparse = utils.random_uniform(
            key, shape=shape, dtype=dtype, fmt=fmt, nnz=nnz
        )
        self.assertArraysEqual(arr, arr_sparse.todense())


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
