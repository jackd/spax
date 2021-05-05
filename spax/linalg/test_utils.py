from functools import partial

import jax
import jax.numpy as jnp
from spax import ops, utils
from spax.linalg.utils import as_array_fun, standardize_signs
from spax.test_utils import random_uniform


@partial(jax.jit, backend="cpu", static_argnums=2)
def eigh_general(A, B, largest: bool):
    if B is None:
        w, v = jnp.linalg.eigh(A)
        B = lambda x: x
    else:
        w, v = jnp.linalg.eig(jnp.linalg.solve(B, A))
        w = w.real
        v = v.real
        i = jnp.argsort(w)
        w = w[i]
        v = v[:, i]
        B = as_array_fun(B)

    if largest:
        w = w[-1::-1]
        v = v[:, -1::-1]

    norm2 = jax.vmap(lambda vi: (vi.conj() @ B(vi)).real, in_axes=1)(v)
    norm = jnp.sqrt(norm2)
    v = v / norm
    v = standardize_signs(v)
    return w, v


def random_symmetric_mat(
    key: jnp.ndarray, size: int, dtype: jnp.dtype = jnp.float32, fmt="dense"
):
    """Get a random symmetric matrix with slightly strengthened diagonal."""
    k0, k1 = jax.random.split(key, 2)
    diag = 1 + jax.random.uniform(k1, (size,), dtype=dtype)
    if fmt == "dense":
        a = jax.random.uniform(k0, (size, size), dtype=dtype)
        a = (a + a.T) / 2
        a = a + jnp.diag(diag)
    else:
        a = random_uniform(k0, (size, size), dtype=dtype, fmt=fmt)
        a = ops.symmetrize(a)
        a = ops.add(a, utils.diag(diag))
    return a
