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

from lowprev_lsip.optimize import MinFun, min_fun_minimize


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
    bounds: Bounds,
    x: npt.NDArray,
) -> float:
    return -min_fun(lambda y: -abs(fun(x) - fun(y)), bounds)


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
        return -modulus_of_continuity_at(
            min_fun_inner, fun, get_neighbourhood_bounds(bounds, x, z), x
        )

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
