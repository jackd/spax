import typing as tp
from functools import partial

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
from jax.config import config

import spax

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
    coords = jnp.asarray(jnp.unravel_index(indices, shape))
    nnz = indices.shape[0]
    data = jax.random.normal(k1, (nnz,), dtype=dtype)
    a = spax.COO(coords, data, shape)
    if fmt == "csr":
        a = a.tocsr()
    else:
        assert fmt == "coo"
    return a


def matmul_benchmark(
    state,
    fmt,
    shape: tp.Tuple[int, int],
    num_vecs: int,
    nnz: tp.Union[int, float],
    backend: tp.Optional[str] = "cpu",
    dtype=jnp.float32,
    seed: int = 0,
):
    ka, kx = jax.random.split(jax.random.PRNGKey(seed))
    a = random_uniform(ka, shape=shape, dtype=dtype, nnz=nnz, fmt=fmt)
    b = jax.random.normal(kx, (shape[1], num_vecs))

    def fun(a, b):
        return spax.ops.matmul(a, b)

    if backend:
        fun = jax.jit(fun, backend=backend)

    fun(a, b).block_until_ready()  # ensure jit has finished
    while state:
        fun(a, b).block_until_ready()


large_kwargs = dict(shape=(int(1e4), int(1e4)), num_vecs=8, nnz=0.01)

for size, size_str in ((large_kwargs, "large"),):
    for dtype, dtype_str in ((jnp.float32, "f32"), (jnp.float64, "f64")):
        for backend in (None, "cpu", "gpu"):
            backend_str = "nojit" if backend is None else backend
            for fmt in "csr", "coo":
                benchmark.register(
                    partial(
                        matmul_benchmark,
                        fmt=fmt,
                        dtype=dtype,
                        backend=backend,
                        **large_kwargs
                    ),
                    name="-".join((size_str, dtype_str, backend_str, fmt)),
                )


if __name__ == "__main__":
    benchmark.main()
