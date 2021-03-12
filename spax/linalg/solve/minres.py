import typing as tp

import jax
import jax.numpy as jnp

from spax.linalg.utils import as_array_fun
from spax.types import ArrayFun, ArrayOrFun


@jax.jit
def sym_ortho(a, b):
    """Algorithm 2 from [choi2012](https://www.mcs.anl.gov/papers/P3027-0812.pdf)."""
    assert isinstance(a, float) or a.dtype in (jnp.float32, jnp.float64)
    assert isinstance(b, float) or b.dtype in (jnp.float32, jnp.float64)
    abs_a = jnp.abs(a)
    abs_b = jnp.abs(b)

    def if_b_0(_):
        # a, b = op
        c = jax.lax.cond(a == 0, lambda _: 1.0, lambda _: jnp.sign(a), None)
        s = 0.0
        r = abs_a
        return c, s, r

    def otherwise(_):
        # b != 0

        def if_a_0(_):
            c = 0.0
            s = jnp.sign(b)
            r = abs_b
            return c, s, r

        def otherwise(_):

            # b != 0, a != 0
            def if_abs_b_greater_equal(_):
                # abs_b >= abs_a
                tau = a / b
                s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
                c = s * tau
                r = b / s
                return c, s, r

            def otherwise(_):
                # abs_a > abs_b
                tau = b / a
                c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
                s = c * tau
                r = a / c
                return c, s, r

            return jax.lax.cond(abs_b >= abs_a, if_abs_b_greater_equal, otherwise, None)

        return jax.lax.cond(a == 0, if_a_0, otherwise, None)

    return jax.lax.cond(b == 0, if_b_0, otherwise, None)


class State(tp.NamedTuple):
    z: float
    q: float
    beta: float
    phi: float
    omega: float
    x: float
    c: jnp.ndarray
    s: jnp.ndarray
    tau: float
    xi: float
    kappa: float
    A: float
    delta: float
    gamma: jnp.ndarray
    eta: float
    nu: float


def _nan_filled(size, dtype=jnp.float32):
    return jnp.full((size,), jnp.nan, dtype=dtype)


class MutableState:
    def __init__(
        self,
        z: tp.Optional[jnp.ndarray] = None,
        q: tp.Optional[jnp.ndarray] = None,
        beta: tp.Optional[float] = None,
        phi: tp.Optional[float] = None,
        omega: tp.Optional[float] = None,
        x: tp.Optional[jnp.ndarray] = None,
        c: tp.Optional[jnp.ndarray] = None,
        s: tp.Optional[jnp.ndarray] = None,
        tau: tp.Optional[float] = None,
        xi: tp.Optional[float] = None,
        kappa: tp.Optional[float] = None,
        A: tp.Optional[float] = None,
        delta: tp.Optional[float] = None,
        gamma: tp.Optional[jnp.ndarray] = None,
        eta: tp.Optional[float] = None,
        nu: tp.Optional[float] = None,
    ):
        self.z = z
        self.q = q
        self.beta = beta
        self.phi = phi
        self.omega = omega
        self.x = x
        self.c = c or _nan_filled(3)
        self.s = s or _nan_filled(3)
        self.tau = tau
        self.xi = xi
        self.kappa = kappa
        self.A = A
        self.delta = delta
        self.gamma = gamma
        self.eta = eta or _nan_filled(2)
        self.nu = nu or _nan_filled(2)

    def to_tuple(self) -> State:
        return State(
            q=self.q,
            beta=self.beta,
            phi=self.phi,
            omega=self.omega,
            x=self.x,
            c=self.c,
            s=self.s,
            tau=self.tau,
            xi=self.xi,
            kappa=self.kappa,
            A=self.A,
            delta=self.delta,
            gamma=self.gamma,
            eta=self.eta,
            nu=self.nu,
        )


def _update(
    a: ArrayFun,
    sigma: float,
    m,
    k: int,
    state: MutableState,
    next_state: MutableState,
    old_states: tp.Tuple[MutableState, ...],  # at least 4
):
    # Algorithm 1: line 8
    state.p = a(state.q) - sigma(state.q)
    state.alpha = jnp.dot(state.q, state.p) / (state.beta * state.beta)
    # Algorithm 1: line 9
    next_state.z = (
        state.p / state.beta
        - state.alpha / state.beta * state.z
        - state.beta / old_states[-1].beta * old_states[-1].z
    )
    # Algorithm 1: line 10
    next_state.q = m(next_state.z)
    next_state.beta = jnp.sqrt(jnp.dot(next_state.z, next_state.q))
    # Algorithm 1: line 11
    state.rho = jnp.linalg.norm(
        (state.alpha, next_state.beta)
        if k == 1
        else (state.beta, state.alpha, next_state.beta)
    )
    # Algorithm 1: line 12
    state.delta[1] = (
        old_states[-1].c[0] * state.delta + old_states[-1].s[0] * state.alpha
    )
    # Algorithm 1: line 13
    state.gamma[0] = (
        old_states[-1].s[0] * state.delta - old_states[-1].c[0] * state.alpha
    )
    # Algorithm 1: line 14
    next_state.epsilon = old_states[-1].s[0] * next_state.beta
    # Algorithm 1: line 15
    next_state.delta = -old_states[-1].c[0] * next_state.beta
    # Algorithm 1: line 16
    c, s, g = sym_ortho(state.gamma, next_state.beta)
    state.c[0] = c
    state.s[0] = s
    state.gamma[1] = g
    # Algorithm 1: line 17
    c, s, g = sym_ortho(old_states[-2].gamma[4], state.epsilon)
    state.c[1] = c
    state.s[1] = g
    old_states[-2].gamma[5] = g
    # Algorithm 1: line 18
    state.delta[2] = state.s[1] * old_states[-1].nu[0] - state.c[1] * state.delta[1]
    state.gamma[2] = -state.c[1] * state.gamma[1]
    # Algorithm 1: line 19
    old_states[-1].nu[1] = (
        state.k[1] * old_states[-1].nu[0] + state.s[1] * state.delta[1]
    )
    # Algorithm 1: line 20
    c, s, g = sym_ortho(old_states[-1].gamma[3], state.delta[2])
    # Algorithm 1: line 21
    state.nu[0] = state.s[2] * state.gamma[2]
    state.gamma[3] = -state.c[2] * state.gamma[2]
    # Algorithm 1: line 22
    state.tau = state.c[0] * old_states[-1].phi
    # Algorithm 1: line 23
    state.phi = state.s[0] * old_states[-1].phi
    old_states.psi = old_states.phi * jnp.linalg.norm((state.gamma, next_state.delta))
    # Algorithm 1: line 24
    state.gamma_min = (
        state.gamma[0]
        if k == 1
        else jnp.min(
            (
                old_states[-1].gamma_min,
                old_states[-2].gamma[5],
                old_states[-1].gamma[4],
                jnp.abs(state.gamma[4]),
            )
        )
    )
    # Algorithm 1: line 25
    state.A = jnp.max(
        (
            old_states[-1].A,
            state.rho,
            old_states[-2].gamma[5],
            old_states[-1].gamma[4],
            jnp.abs(state.gamma[3]),
        )
    )
    # Algorithm 1: line 26
    state.omega = jnp.linalg.norm((old_states[-1].omega, state.tau))
    state.kappa = state.A / state.gamma_min
    # Algorithm 1: line 27
    state.w[0] = -state.c[1] / state.beta * state.q + state.s[1] * old_states[-2].w[2]
    # Algorithm 1: line 28
    old_states[-2].w[3] = (
        state.s[1] / state.beta * state.q + state.c[1] * old_states[-2].w[2]
    )
    # Algorithm 1: line 29
    if k > 2:
        state.w[1] = (
            state.s[2] * old_states[-1] * old_states[-1].w[1] - state.c[2] * state.w[0]
        )
    old_states[-1].w[2] = state.c[2] * old_states[-1].w[1] + state.s[2] * state.w[0]
    # Algorithm 1: line 30
    if k > 2:
        old_states[-1].mu[2] = (
            old_states[-2].tau
            - old_states[-2].eta[0] * old_states[-4].mu[3]
            - old_states[-2].nu[0] * old_states[-3].mu[2]
        ) / old_states[-2].gamma[5]

    # Algorithm 1: line 31
    if k > 1:
        old_states[-1].mu[1] = (
            old_states[-1].tau
            - old_states[-1].eta[0] * old_states[-3].mu[2]
            - old_states[-1].nu[0] * old_states[-2].mu[2]
        ) / old_states[-1].gamma[4]
    # Algorithm 1: line 32
    state.mu[0] = jax.lax.cond(
        state.gamma[3] == 0,
        lambda _: 0,
        lambda _: (
            state.tau
            - state.eta[0] * old_states[-2].mu[2]
            - state.nu[0] * old_states[-1].eta[1]
        )
        / state.gamma[3],
    )
    # Algorithm 1: line 33
    old_states[-2].x[1] = (
        old_states[-3].x[1] + old_states[-2].mu[2] * old_states[-2].w[1]
    )
    # Algorithm 1: line 34
    state.x = (
        old_states[-2].x[1]
        + old_states[-1].mu[1] * old_states[-1].w[2]
        + state.mu * state.w[1]
    )
    # Algorithm 1: line 35
    old_states[-2].xi[1] = jnp.linalg.norm((old_states[-2].xi[1], old_states[-2].mu[2]))
    # Algorithm 1: line 36
    state.xi[0] = jnp.linalg.norm(
        (old_states[-2].xi[1], old_states[-1].mu[1], state.mu[0])
    )


def minres_qlp(
    A: ArrayOrFun,
    b: jnp.ndarray,
    sigma: float = 0.0,
    iK: tp.Optional[ArrayOrFun] = None,
):
    """
    Solve `(A - sigma*I) x = b` for symmetric `A`.

    If `A - sigma * I` is singular Returns least squares solution.

    This is Algorithm 1 in [choi2012](https://www.mcs.anl.gov/papers/P3027-0812.pdf).

    ```bibtex
    @techreport{choi2012algorithm,
        title={ALGORITHM \& DOCUMENTATION: MINRESQLP for singular symmetric and
               Hermitian linear equations and least-squares problems},
        author={Choi, Sou-Cheng T and Saunders, Michael A},
        year={2012},
        institution={Technical Report ANL/MCS-P3027-0812, Computation Institute,
                     University of~…}
    }
    ```

    Args:
        A: array or callable implementing symmetric matrix multiplication.
        b: rhs array
        sigma: float, see formula
        iK: callable representing preconditioning inverse, inv(M) from choi2012. Uses
            identity if not given.
    """
    A = as_array_fun(A)
    iK = lambda x: x if iK is None else as_array_fun(iK)

    z1 = b
    q1 = iK(z1)

    jnp.sqrt(b @ q1)
    wn = w0 = 0
    xn2 = xn1 = x0 = 0
    c01 = c02 = c03 = -1
    s01 = s02 = s03 = -1
    t0 = omega0 = xin2 = xin1 = x0 = 0
    A0 = delta0 = gamman1 = gamma0 = etan1 = eta0 = eta1 = v0 = v1 = mun1 = mu0 = 0
