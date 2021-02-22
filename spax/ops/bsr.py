import jax
import jax.numpy as jnp

from spax.sparse import BSR


def matmul(a: BSR, v) -> jnp.ndarray:
    v = jnp.asarray(v)
    trailing = v.shape[1:]
    v = v.reshape(-1, a.blocksize[1], *trailing)
    dv = jax.vmap(jnp.dot)(a.data, v[a.indices])
    ind = jnp.cumsum(jnp.zeros_like(a.indices).at[a.indptr].add(1))
    return (
        jnp.zeros((a.blockshape[0], a.blocksize[0], *trailing), dv.dtype)
        .at[ind - 1]
        .add(dv)
        .reshape(-1, *trailing)
    )
