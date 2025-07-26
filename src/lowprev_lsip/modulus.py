"""
Modulus
=======

Calculate modulus of continuity under the max norm,
for functions with a rectangular domain.
"""

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

from lowprev_lsip.optimize import (
    MinFun,
    max_fun,
)


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
    """Calculate bounds for a neighbourhood at *x* of size *z* under the max norm."""
    lb = np.max([bounds.lb, x - z], axis=0)
    ub = np.min([bounds.ub, x + z], axis=0)
    return Bounds(lb, ub)


def modulus_of_continuity_at(
    min_fun: MinFun,
    fun: Callable[[npt.NDArray], float],
    x: npt.NDArray,
) -> float:
    return max_fun(min_fun, lambda y: abs(fun(x) - fun(y)))[0]


def modulus_of_continuity(
    fun: Callable[[npt.NDArray], float],
    bounds: Bounds,
    z: float,
    min_fun: Callable[[Bounds], MinFun],
    min_fun_inner: Callable[[Bounds], MinFun],
) -> float:
    """Calculate modulus of continuity.

    This function uses a nested optimization strategy.
    The *min_fun* routing runs an optimization over the entire domain.
    For each point in the domain, *min_fun_inner* runs an optimization over its
    neighbourhood.
    """

    def fun2(x: npt.NDArray) -> float:
        return modulus_of_continuity_at(
            min_fun_inner(get_neighbourhood_bounds(bounds, x, z)), fun, x
        )

    return max_fun(min_fun(bounds), fun2)[0]


def lipschitz_constant(
    fun_grad: Callable[[npt.NDArray], npt.NDArray],
    min_fun: MinFun,
) -> float:
    """Calculate the Lipschitz constant, from the given function gradient.
    The operator norm is the L1 norm, which is induced by the max norm.
    """

    def fun2(x: npt.NDArray) -> float:
        return np.sum(np.abs(fun_grad(x)))

    return max_fun(min_fun, fun2)[0]
