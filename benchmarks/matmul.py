import typing as tp
from functools import partial

import google_benchmark as benchmark

import jax
import jax.numpy as jnp
import spax
from jax.config import config
from jax.experimental.sparse_ops import COO

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def random_uniform(key, shape, dtype, nnz, fmt):
    """Similar to spax.utils.random_uniform, but can generate way larger matrices."""
    assert len(shape) == 2
    num_elements = jnp.prod(jnp.asarray(shape))
    if 0 < nnz < 1:
        nnz = int(nnz * num_elements)
    assert isinstance(nnz, int)

    k0, k1 = jax.random.split(key)
    indices = jax.random.uniform(
        k0, (nnz,), dtype=jnp.float32, maxval=num_elements
    ).astype(jnp.int32)
    indices = jnp.unique(indices)
    row, col = jnp.unravel_index(indices, shape)
    nnz = indices.shape[0]
    data = jax.random.normal(k1, (nnz,), dtype=dtype)
    a = COO((data, row, col), shape=shape)
    if fmt == "csr":
        a = spax.ops.to_csr(a)
    else:
        assert fmt == "coo"
    return a


def matmul_benchmark(
    state,
    fmt,
    shape: tp.Tuple[int, int],
    num_vecs: int,
    nnz: tp.Union[int, float],
    backend: str = "cpu",
    dtype=jnp.float32,
    seed: int = 0,
):
    ka, kx = jax.random.split(jax.random.PRNGKey(seed))
    a = random_uniform(ka, shape=shape, dtype=dtype, nnz=nnz, fmt=fmt)
    b = jax.random.normal(kx, (shape[1], num_vecs), dtype=dtype)

    @partial(jax.jit, backend=backend)
    def fun(a, b):
        return a @ b

    fun(a, b).block_until_ready()  # ensure jit has finished
    while state:
        fun(a, b).block_until_ready()


large_kwargs = dict(shape=(int(1e4), int(1e4)), num_vecs=8, nnz=0.01)

for size, size_str in ((large_kwargs, "large"),):
    for dtype, dtype_str in ((jnp.float32, "f32"), (jnp.float64, "f64")):
        for backend in ("cpu", "gpu"):
            for fmt in "csr", "coo":
                benchmark.register(
                    partial(
                        matmul_benchmark,
                        fmt=fmt,
                        dtype=dtype,
                        backend=backend,
                        **large_kwargs
                    ),
                    name="-".join((size_str, dtype_str, backend, fmt)),
                )


if __name__ == "__main__":
    benchmark.main()
