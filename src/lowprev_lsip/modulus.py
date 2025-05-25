"""
Modulus
=======

Calculate modulus of continuity under the max norm,
for functions with a rectangular domain.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, brute, minimize


def modulus_of_continuity_slow(
    fun: Callable[[npt.NDArray], float],
    points: Sequence[npt.NDArray],
    z: float,
):
    """Slow reference implementation."""
    return max(
        (abs(fun(x) - fun(y)), x, y)
        for x in points
        for y in points
        if np.max(np.abs(x - y)) <= z
    )


def get_neighbourhood_bounds(
    bounds: Bounds,
    x: npt.NDArray,
    z: float,
) -> Bounds:
    lb = np.max([bounds.lb, x - z], axis=0)
    ub = np.min([bounds.ub, x + z], axis=0)
    return Bounds(lb, ub)


class MinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> float: ...


def modulus_of_continuity_at(
    min_fun: MinFun,
    fun: Callable[[npt.NDArray], float],
    bounds: Bounds,
    x: npt.NDArray,
    z: float,
) -> float:
    bounds2 = get_neighbourhood_bounds(bounds, x, z)
    return -min_fun(lambda y: -abs(fun(x) - fun(y)), bounds2)


def modulus_of_continuity(
    fun: Callable[[npt.NDArray], float],
    bounds: Bounds,
    z: float,
    min_fun: MinFun | None = None,
    min_fun_inner: MinFun | None = None,
) -> float:
    min_fun = min_fun if min_fun is not None else min_fun_minimize
    min_fun_inner = min_fun_inner if min_fun_inner is not None else min_fun_minimize

    def fun2(x: npt.NDArray) -> float:
        return -modulus_of_continuity_at(min_fun_inner, fun, bounds, x, z)

    return -min_fun(fun2, bounds)


def lipschitz_constant(
    fun_grad: Callable[[npt.NDArray], npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun | None = None,
) -> float:
    """Calculate the Lipschitz constant, from the given function gradient.
    The operator norm is the L1 norm, which is induced by the max norm.
    """
    min_fun = min_fun if min_fun is not None else min_fun_minimize

    def fun2(x: npt.NDArray) -> float:
        return -np.sum(np.abs(fun_grad(x)))

    return -min_fun(fun2, bounds)


def min_fun_minimize(fun: Callable[[npt.NDArray], float], bounds: Bounds) -> float:
    """Wrapper around :func:`scipy.optimize.minimize`.
    Suitable for convex functions.
    """
    return minimize(fun, 0.5 * (bounds.lb + bounds.ub), bounds=bounds).fun


def min_fun_brute(fun: Callable[[npt.NDArray], float], bounds: Bounds, ns=20) -> float:
    """Wrapper around :func:`scipy.optimize.brute`.
    Finishes with `scipy.optimize.minimize` to respect the bounds.
    Suitable for non-linear functions.
    """

    # note: scipy-stubs has too limited type checking for finish
    def finish(fun2, x0, args, **kwargs):
        return minimize(fun2, x0, args=args, bounds=bounds, **kwargs)

    _, val_opt, _, _ = brute(
        fun, tuple(zip(bounds.lb, bounds.ub)), Ns=ns, full_output=True, finish=finish
    )
    return val_opt
