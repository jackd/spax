import abc
import itertools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.ops import COO, JAXSparse


def _is_complex_dtype(dtype: jnp.dtype) -> bool:
    return dtype in (jnp.complex64, jnp.complex128)  # TODO: better test?


class LinearOperator(abc.ABC):
    @abc.abstractmethod
    def tree_flatten(self):
        raise NotImplementedError("Abstract method")

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @abc.abstractmethod
    def _matmul(self, x):
        raise NotImplementedError("Abstract method")

    @abc.abstractproperty
    def dtype(self) -> jnp.dtype:
        raise NotImplementedError("Abstract property")

    @abc.abstractproperty
    def shape(self) -> tp.Tuple[int, ...]:
        raise NotImplementedError("Abstract property")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_self_adjoint(self) -> bool:
        return False

    def __add__(self, other):
        assert isinstance(other, LinearOperator)
        return Sum(self, other)

    def __matmul__(self, x):
        if isinstance(x, jnp.ndarray):
            return self._matmul(x)
        if isinstance(x, JAXSparse):
            x = MatrixWrapper(x)
        if isinstance(x, LinearOperator):
            return Product(self, x)
        raise TypeError(f"Invalid type of x {x}")

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            return scale(self, x)
        if isinstance(x, jnp.ndarray):
            return Product(self, Diag(x))
        raise ValueError(f"Only multiplication by int / float supported, got {x}")

    @property
    def adjoint(self) -> "LinearOperator":
        if self.is_self_adjoint:
            return self
        raise NotImplementedError(f"adjoint for class {type(self)} not implemented")


@jax.tree_util.register_pytree_node_class
class MatrixWrapper(LinearOperator):
    def __init__(self, A, *, is_self_adjoint: bool = False):
        for attr in "__matmul__", "shape", "dtype":
            assert hasattr(A, attr), attr
        self.A = A
        self._is_self_adjoint = is_self_adjoint

    @property
    def is_self_adjoint(self) -> bool:
        return self._is_self_adjoint

    def tree_flatten(self):
        return (self.A,), {"is_self_adjoint": self._is_self_adjoint}

    @property
    def dtype(self) -> jnp.dtype:
        return self.A.dtype

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self.A.shape

    def _matmul(self, x):
        return self.A @ x


@jax.tree_util.register_pytree_node_class
class SelfAdjointInverse(LinearOperator):
    """Solution to a self-adjoint linear system using conjugate gradient."""

    def __init__(self, A: LinearOperator, **kwargs):
        assert isinstance(A, LinearOperator), type(A)
        assert A.is_self_adjoint
        self.A = A
        self._cg_kwargs = kwargs

    def __repr__(self):
        return f"SelfAdjointInverse({self.A})"

    def tree_flatten(self):
        return (self.A,), self._cg_kwargs

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return self.A.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.A.dtype

    @jax.jit
    def _matmul(self, x):
        assert self.shape[1] == x.shape[0], (self.shape, x.shape)
        out = jax.scipy.sparse.linalg.cg(lambda x: self.A @ x, x, **self._cg_kwargs)[0]
        return out

    @property
    def is_self_adjoint(self):
        return True


@jax.tree_util.register_pytree_node_class
class MappedMatmul(LinearOperator):
    """
    Object that performs `__matmul__` using `jax.lax.map`.

    This won't be as fast as `vmap` but has reduced memory usage.
    """

    def __init__(self, A, *, is_self_adjoint: bool = False):
        assert hasattr(A, "__matmul__"), A
        self.A = A
        self._is_self_adjoint = is_self_adjoint

    @property
    def is_self_adjoint(self) -> bool:
        return self._is_self_adjoint

    def tree_flatten(self):
        return (self.A,), {"is_self_adjoint": bool}

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        return self.A.dtype

    def _matmul(self, x):
        if x.ndim == 1:
            return self.A @ x
        if x.ndim == 2:
            return jax.lax.map(lambda xi: self.A @ xi, x.T).T
        raise ValueError(f"x must be rank 1 or 2 but has shape {x.shape}")


@jax.tree_util.register_pytree_node_class
class Product(LinearOperator):
    def __init__(self, *args):
        assert len(args) > 0
        args = tuple(
            itertools.chain(
                *(arg.factors if isinstance(arg, Product) else (arg,) for arg in args)
            )
        )
        for i, arg in enumerate(args):
            assert hasattr(arg, "__matmul__"), i
        for i in range(len(args) - 1):
            assert args[i].shape[-1] == args[i + 1].shape[0], (
                i,
                args[i].shape,
                args[i + 1].shape,
            )
            assert args[i + 1].dtype == args[0].dtype
        self.factors = args

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert not aux_data
        return Product(*children)

    def tree_flatten(self):
        return self.factors, None

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        shape = list(self.factors[0].shape[:-1])
        for arg in self.factors[1:-1]:
            shape.extend(arg.shape[1:-1])
        shape.extend(self.factors[-1].shape[1:])
        return tuple(shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self.factors[0].dtype

    def _matmul(self, x):
        for factor in self.factors[-1::-1]:
            x = factor @ x
        return x

    @property
    def adjoint(self) -> LinearOperator:
        return Product(*(factor.adjoint for factor in self.factors[-1::-1]))


@jax.tree_util.register_pytree_node_class
class Sum(LinearOperator):
    def __init__(self, *args, is_self_adjoint: tp.Optional[bool] = None):
        assert len(args) > 0
        self._is_self_adjoint = is_self_adjoint
        args = tuple(
            itertools.chain(
                *(arg.terms if isinstance(arg, Sum) else (arg,) for arg in args)
            )
        )
        for i, arg in enumerate(args):
            assert hasattr(arg, "__matmul__"), i
            assert arg.dtype == args[0].dtype, (i, arg.dtype, args[0].dtype)
            assert arg.shape == args[0].shape, (i, arg.shape, args[0].shape)
        self.terms = args
        assert len(args) > 0

    @property
    def is_self_adjoint(self) -> bool:
        if self._is_self_adjoint is None:
            self._is_self_adjoint = all(term.is_self_adjoint for term in self.terms)
        return self._is_self_adjoint

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return Sum(*children, **aux_data)

    def tree_flatten(self):
        return self.terms, dict(is_self_adjoint=self._is_self_adjoint)

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self.terms[0].shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.terms[0].dtype

    def _matmul(self, x):
        return sum((term @ x for term in self.terms))

    def __repr__(self):
        return f"Sum({self.terms})"

    @property
    def adjoint(self) -> LinearOperator:
        if self.is_self_adjoint:
            return self
        return Sum(*(term.adjoint for term in self.terms), is_self_adjoint=False)


@jax.tree_util.register_pytree_node_class
class Diag(LinearOperator):
    def __init__(self, v: jnp.ndarray):
        assert v.ndim == 1
        self.v = v

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert len(children) == 1
        assert not aux_data
        return Diag(*children)

    def tree_flatten(self):
        return (self.v,), None

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return (self.v.size,) * 2

    @property
    def dtype(self) -> jnp.dtype:
        return self.v.dtype

    def _matmul(self, x):
        if x.ndim == 1:
            return self.v * x
        if x.ndim == 2:
            return self.v[:, jnp.newaxis] * x
        raise ValueError(f"x must be rank 1 or 2 but has shape {x.shape}")

    @property
    def is_self_adjoint(self) -> bool:
        return not _is_complex_dtype(jnp.dtype)

    @property
    def adjoint(self) -> bool:
        return Diag(jnp.conj(self.v))


@jax.tree_util.register_pytree_node_class
class Identity(LinearOperator):
    def __init__(self, *, n: int, dtype: tp.Optional[jnp.ndarray] = None):
        self._n = n
        self._dtype = dtype

    def tree_flatten(self):
        return (), {"n": self._n, "dtype": self._dtype}

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    def _matmul(self, x):
        assert self._n == x.shape[0], (self._n, x.shape)
        if self.dtype:
            assert self.dtype == x.dtype, (self.dtype, x.dtype)
        return x

    def __matmul__(self, x):
        return self._matmul(x)

    def __repr__(self):
        return f"Identity({self._n}, {self._dtype})"

    @property
    def is_self_adjoint(self) -> bool:
        return True


@jax.tree_util.register_pytree_node_class
class Scale(LinearOperator):
    def __init__(
        self,
        scale: tp.Union[int, float, jnp.ndarray],
        *,
        size: int,
    ):
        self._scale = scale
        self._size = size

    def tree_flatten(self):
        return (self._scale,), {"size": self._size}

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return (self._size,) * 2

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> tp.Optional[jnp.dtype]:
        return self._scale.dtype

    def _matmul(self, x):
        assert x.shape[0] == self._size
        return self._scale * x

    @property
    def is_self_adjoint(self):
        return isinstance(self._scale, (int, float)) or not _is_complex_dtype(
            self._scale.dtype
        )


@jax.tree_util.register_pytree_node_class
class Take(LinearOperator):
    def __init__(
        self,
        indices: jnp.ndarray,
        *,
        input_size: int,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        self._input_size = input_size
        self._indices = indices
        self._dtype = dtype

    def tree_flatten(self):
        return (self._indices,), {"input_size": self._input_size, "dtype": self._dtype}

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return (self._indices.size, self._input_size)

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    def _matmul(self, x):
        assert x.shape[0] == self._input_size
        return x[self._indices]


@jax.tree_util.register_pytree_node_class
class HStacked(LinearOperator):
    def __init__(self, arg, *args):
        assert len(arg.shape) == 2
        assert all(arg.shape[:-1] == arg_.shape[:-1] for arg_ in args)
        assert all(arg_.dtype == arg.dtype for arg_ in args)
        self._args = (arg, *args)
        self._sizes = np.asarray([arg.shape[1] for arg in self._args])
        self._sections = np.cumsum(self._sizes)

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return (self._args[0].shape[0], self._sections[-1])

    @property
    def dtype(self):
        return self._args[0].dtype

    @property
    def args(self):
        return self._args

    def tree_flatten(self):
        return self._args, None

    def _matmul(self, x):
        return sum(
            (arg @ xi for arg, xi in zip(self._args, jnp.split(x, self._sections[:-1])))
        )


def take(A, indices):
    return Product(Take(indices, input_size=A.shape[0], dtype=A.dtype), A)


def scale(A, scalar: tp.Union[int, float, jnp.ndarray]):
    if isinstance(scalar, int):
        assert jnp.issubdtype(A.dtype, jnp.integer)
        scalar = jnp.asarray(scalar, A.dtype)
    elif isinstance(scalar, float):
        assert jnp.issubdtype(A.dtype, jnp.floating)
        scalar = jnp.asarray(scalar, A.dtype)
    else:
        assert isinstance(scalar, jnp.ndarray)
        assert scalar.dtype == A.dtype, (scalar.dtype, A.dtype)
    return Product(Scale(scalar, size=A.shape[0]), A)


def identity_plus(A):
    return Sum(Identity(n=A.shape[0], dtype=A.dtype), A)


def symmetric_inverse(
    operator: tp.Union[jnp.ndarray, JAXSparse, LinearOperator], **cg_kwargs
) -> LinearOperator:
    """Assumes operator is symmetric or a product of symmetric factors."""
    assert operator.ndim == 2, operator.ndim
    assert operator.shape[0] == operator.shape[1], operator.shape
    if isinstance(operator, SelfAdjointInverse):
        return operator.A
    if isinstance(operator, Diag):
        return Diag(1 / operator.v)
    if isinstance(operator, Identity):
        return operator
    if isinstance(operator, Product):
        return Product(
            *(symmetric_inverse(f, **cg_kwargs) for f in operator.factors[-1::-1])
        )
    return SelfAdjointInverse(operator, **cg_kwargs)


def scatter_limit_split(mat: COO, is_self_adjoint: bool = False) -> LinearOperator:
    """
    Get a linear operator that overcomes the scatter limit.

    By default `jax.lax.scatter` based ops (including `COO.__matmul__`) are limited to
    indices with size <= np.iinfo(np.int32).max == 2147483647. This overcomes that by
    creating a linear operator consisting of a sum.
    """
    size = mat.data.size
    max_size = np.iinfo(np.int32).max
    if size <= max_size:
        return MatrixWrapper(mat, is_self_adjoint=is_self_adjoint)
    leading = size - size % max_size
    data, data_rem = jnp.split(mat.data, (leading,))
    row, row_rem = jnp.split(mat.row, (leading,))
    col, col_rem = jnp.split(mat.col, (leading,))

    shape = mat.shape
    num_splits = size // max_size

    terms = [
        COO((d, r, c), shape=shape)
        for d, r, c in zip(
            jnp.split(data, num_splits),
            jnp.split(row, num_splits),
            jnp.split(col, num_splits),
        )
    ]
    terms.append(COO((data_rem, row_rem, col_rem), shape=shape))
    print(f"scatter_limit_split gives {len(terms)}-term sum")
    return Sum(*terms, is_self_adjoint=is_self_adjoint)
