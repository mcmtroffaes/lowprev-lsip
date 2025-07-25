from collections.abc import Callable
from typing import Protocol

import numpy.typing as npt
from scipy.optimize import Bounds, brute, minimize


class MinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> tuple[npt.NDArray, float]: ...


def max_fun(
    min_fun: MinFun, fun: Callable[[npt.NDArray], float], bounds: Bounds
) -> tuple[npt.NDArray, float]:
    x_star, f_star = min_fun(lambda x: -fun(x), bounds)
    return x_star, -f_star


def min_fun_minimize(
    fun: Callable[[npt.NDArray], float], bounds: Bounds
) -> tuple[npt.NDArray, float]:
    """Wrapper around :func:`scipy.optimize.minimize`.
    Suitable for convex functions.
    """
    result = minimize(fun, 0.5 * (bounds.lb + bounds.ub), bounds=bounds)
    return result.x, result.fun


def min_fun_brute(ns=20) -> MinFun:
    """Wrapper around :func:`scipy.optimize.brute`.
    Finishes with `scipy.optimize.minimize` to respect the bounds.
    Suitable for non-linear functions.
    """

    def _(
        fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> tuple[npt.NDArray, float]:
        # note: scipy-stubs has too limited type checking for finish
        def finish(fun2, x0, args, **kwargs):
            return minimize(fun2, x0, args=args, bounds=bounds, **kwargs)

        x_star, fun_star, _, _ = brute(
            fun,
            tuple(zip(bounds.lb, bounds.ub)),
            Ns=ns,
            full_output=True,
            finish=finish,
        )
        return x_star, fun_star

    return _
